import os
import io
from pathlib import Path
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from taxonova.model.model_factory import ModelFactory
from taxonova.model.model_config import ModelConfig
from taxonova.preprocessing.sequence_processor import SequenceProcessor
from taxonova.data.dataset import TaxonomyDataset
from taxonova.data.fasta import parse_sequence_id, parse_taxonomy

app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

@app.on_event("startup")
async def load_model():
    model_path = os.getenv("TAXONOVA_MODEL_PATH", "models/my_model.pt")
    model, taxonomy_info = ModelFactory.load_model(model_path)

    if 'kmer_values' in taxonomy_info:
        processor = SequenceProcessor(k=taxonomy_info['kmer_values'])
    else:
        k = int(np.log(taxonomy_info['input_dim']) / np.log(4))
        processor = SequenceProcessor(k=k)

    app.state.model = model
    app.state.taxonomy_info = taxonomy_info
    app.state.processor = processor

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def read_fasta_string(text: str):
    sequences = []
    seq_id = None
    header = None
    seq_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if seq_id is not None:
                seq = ''.join(seq_lines)
                taxonomy = parse_taxonomy(header, expect_taxonomy=False)
                sequences.append((seq_id, seq, taxonomy))
            header = line
            seq_id = parse_sequence_id(line)
            seq_lines = []
        else:
            seq_lines.append(line)
    if seq_id is not None:
        seq = ''.join(seq_lines)
        taxonomy = parse_taxonomy(header, expect_taxonomy=False)
        sequences.append((seq_id, seq, taxonomy))
    return sequences

def prepare_plain_sequences(text: str):
    seqs = []
    for idx, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if line:
            seqs.append((f"seq{idx}", line, {"domain":"unknown","phylum":"unknown","class":"unknown","order":"unknown","family":"unknown","genus":"unknown","species":"unknown"}))
    return seqs

@app.post("/classify")
async def classify(sequences: str = Form(None), file: UploadFile = File(None), v_region: str = Form(None)):
    seq_data = []
    if file is not None:
        content = await file.read()
        seq_text = content.decode()
        if '>' in seq_text:
            seq_data.extend(read_fasta_string(seq_text))
        else:
            seq_data.extend(prepare_plain_sequences(seq_text))
    if sequences:
        if '>' in sequences:
            seq_data.extend(read_fasta_string(sequences))
        else:
            seq_data.extend(prepare_plain_sequences(sequences))
    if not seq_data:
        return {"error": "No sequences provided"}

    taxonomy_info = app.state.taxonomy_info
    processor = app.state.processor
    dataset = TaxonomyDataset(
        sequences=seq_data,
        processor=processor,
        label_encoder=taxonomy_info.get('label_encoder'),
        domain_encoder=taxonomy_info.get('domain_encoder'),
        phylum_encoder=taxonomy_info.get('phylum_encoder'),
        class_encoder=taxonomy_info.get('class_encoder'),
        order_encoder=taxonomy_info.get('order_encoder'),
        family_encoder=taxonomy_info.get('family_encoder'),
        genus_encoder=taxonomy_info.get('genus_encoder'),
        species_encoder=taxonomy_info.get('species_encoder'),
        taxonomy_hierarchy=taxonomy_info.get('taxonomy_hierarchy')
    )

    config = ModelConfig(input_dim=taxonomy_info['input_dim'])
    trainer = ModelFactory.create_trainer(config=config, dataset=dataset, model=app.state.model)
    preds = trainer.predict(seq_data)

    results = []
    for i, (seq_id, _, _) in enumerate(seq_data):
        entry = {"sequence_id": seq_id}
        for level in trainer.level_order:
            entry[level] = preds['predictions'][level][i]
            entry[f"{level}_confidence"] = float(preds['confidence'][level][i])
        results.append(entry)

    # Build TSV string
    tsv_io = io.StringIO()
    header = ["sequence_id"]
    for level in trainer.level_order:
        header.extend([level, f"{level}_confidence"])
    tsv_io.write("\t".join(header) + "\n")
    for r in results:
        row = [r["sequence_id"]]
        for level in trainer.level_order:
            row.extend([r[level], f"{r[level+'_confidence']:.4f}"])
        tsv_io.write("\t".join(row) + "\n")
    return {"results": results, "tsv": tsv_io.getvalue()}
