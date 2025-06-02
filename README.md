# ARJudge

[Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework](https://arxiv.org/abs/2502.18874) (ACL 2025 Findings)

ARJudge is a unified and robust framework for multi-faceted evaluation of language model responses. The system learns to align evaluation across different criteria by generating instruction-specific evaluation questions and providing detailed analysis before making final judgments.

## 🚀 Quick Start

### Prerequisites

- vllm=0.5.4
- transformers=4.44.2
- trl=0.9.6
- [alignment-handbook](https://github.com/huggingface/alignment-handbook)

### Download Training Data

Download the Composite Analysis Corpus ([Google Drive](https://drive.google.com/file/d/1pjT0d_TYSzTA90Fbbq-BtVMSCN8-HglS/view?usp=sharing))

### Download Our Model

| Model Name | HF Checkpoint | Size | License |
|------------|---------------|------|---------|
| ARJudge | [kaishxu/ARJudge](https://huggingface.co/kaishxu/ARJudge) | 7B | [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE) |

## 📖 Usage

### Training

Our model is based on [Qwen2.5-7B-Instruct](). Please donwload the model from Hugging Face. Train the ARJudge model using the provided training script:

```bash
# Configure training parameters in configs/config.yaml
# Then run training
bash scripts/train.sh
```

The training script will:
- Load the Qwen2.5-7B-Instruct base model
- Fine-tune on the ARJudge training data
- Save checkpoints to the specified output directory

### Evaluation

Evaluate the trained model on test datasets:

```bash
# Run evaluation on Auto-J dataset
bash scripts/eval.sh
```

The evaluation pipeline includes:
1. **Analysis Generation**: Generate multi-faceted analysis using the trained ARJudge model
2. **Judgment Refinement**: Use a base model to make final comparative judgments
3. **Metrics Calculation**: Compute evaluation metrics

## 🔧 Configuration

### Training Configuration

Key parameters in `configs/config.yaml`:

```yaml
# Model settings
model_name_or_path: /path/to/Qwen2.5-7B-Instruct
torch_dtype: bfloat16
attn_implementation: flash_attention_2

# Training settings
learning_rate: 1.0e-05
num_train_epochs: 2
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
max_seq_length: 4096
```

### Evaluation Settings

Modify the evaluation datasets as: Auto-J, MTBench, etc.

## 📊 Data Format

### Training Data Format

```json
{
  "prompt": "instruction",
  "output": "response"
}
```

### Test Data Format

```json
{
  "idx": 0,
  "instruction": "instruction",
  "response1": "response1",
  "response2": "response2", 
  "winner": 1 // or 2
}
```