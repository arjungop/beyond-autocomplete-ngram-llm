# Beyond Autocomplete: N‑Gram Language Models for Sentence Completion and Synthetic Corpus Generation

**Advanced N-gram language models for sentence completion, synthetic corpus generation, and LLM training under data scarcity conditions**

[![Language](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Paper%20Included-orange.svg)](research_paper.md)
[![BLEU](https://img.shields.io/badge/BLEU%20Score-0.275-brightgreen.svg)](eval_metrics.py)

## Overview

This project implements a comprehensive **statistical N-gram language model** specifically designed for **autocomplete and word prediction** applications. The system uses 1-gram through 4-gram models with optimized smoothing techniques, achieving **BLEU scores of 0.275** and providing real-time interactive word suggestions.

### Key Achievements
- **BLEU Score**: 0.275 average (Peak: 0.548)
- **Performance**: 4-gram models achieve best results (0.369 BLEU)
- **Application**: Real-time autocomplete system
- **Documentation**: Complete research paper with methodology
- **Evaluation**: Dual-metric assessment (Perplexity + BLEU)

## Features

### Interactive Autocomplete System
- **Real-time word prediction** as you type
- **Probability-ranked suggestions** with confidence scores
- **Context-aware completions** using N-gram context
- **Social media adaptation** trained on Twitter data

### Comprehensive Evaluation
- **BLEU Score Assessment**: Standardized text generation quality metrics
- **Perplexity Evaluation**: Traditional language modeling metrics  
- **Method Comparison**: Performance across different N-gram orders
- **Quantitative Analysis**: Statistical validation of results

### Text Generation
- **Markov Chain Generator**: Creates realistic social media text
- **Interactive Interface**: Customizable text length and style
- **Twitter-style Output**: Captures contemporary language patterns

### Research-Grade Implementation
- **Academic Documentation**: Complete methodology and results
- **Mathematical Foundations**: Detailed algorithmic descriptions
- **Reproducible Results**: Fixed seeds and documented parameters

## Quick Start

### Prerequisites
```bash
# Python 3.7+ required
python --version
```

### Installation
```bash
# Clone the repository
git clone https://github.com/arjungop/beyond-autocomplete-ngram-llm.git
cd beyond-autocomplete-ngram-llm

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Run Interactive Autocomplete
```bash
python main.py
```
**Example Session:**
```
Enter a sentence (or 'q' to quit): I am going to
Suggestions:
1. the (probability: 0.234567)
2. be (probability: 0.189234)
3. work (probability: 0.156789)
```

### Run BLEU Evaluation
```bash
python eval_metrics.py
```
**Sample Output:**
```
AUTOCOMPLETE BLEU EVALUATION
Average BLEU Score: 0.275
Maximum BLEU Score: 0.548
Performance Level: Good
Best Method: 4-gram (k=0.005) (BLEU: 0.369)
```

### Generate Text
```bash
python text_generation_demo.py
```

## 📁 Project Structure

### 🔧 Core System
| File | Purpose | Description |
|------|---------|-------------|
| **`main.py`** | 🎮 **Interactive Interface** | Real-time autocomplete with user input |
| **`eval_metrics.py`** | 📊 **BLEU Evaluation** | Comprehensive quality assessment |
| **`ngram_model.py`** | 🧠 **Advanced Models** | Optimized N-gram implementation |
| **`language_model.py`** | ⚙️ **Core Engine** | Probability estimation and prediction |
| **`data_preprocessing.py`** | 🔄 **Data Pipeline** | Text cleaning and tokenization |

### 📝 Text Generation
| File | Purpose | Description |
|------|---------|-------------|
| **`markov_text_generator.py`** | 🎨 **Text Generator** | Markov chain implementation |
| **`text_generation_demo.py`** | 🎪 **Generation Interface** | Interactive text creation |

### 📚 Documentation
| File | Purpose | Description |
|------|---------|-------------|
| **`research_paper.md`** | 📖 **Academic Paper** | Complete methodology and results |
| **`README.md`** | 📋 **Project Guide** | This comprehensive overview |

### 📦 Data & Resources
| Directory | Contents |
|-----------|----------|
| **`data/`** | Training corpus (Twitter dataset) |
| **`archived_files/`** | Legacy versions and development files |

## Usage Examples

### Interactive Autocomplete
```python
# System automatically loads and provides suggestions
python main.py

# User types: "thank you for"
# System suggests: "sharing" (32.1%), "the" (28.9%), "your" (21.3%)
```

### Programmatic Usage
```python
from ngram_model import NGramModel
from data_preprocessing import get_tokenized_data

# Load and train model
model = NGramModel(max_n=4)
model.train(training_data, count_threshold=1)

# Get predictions
suggestions = model.get_suggestions("thank you for", num_suggestions=5)
```

### BLEU Evaluation
```python
from eval_metrics import calculate_bleu

# Evaluate completion quality
bleu_score = calculate_bleu(
    candidate="i am going to work today",
    reference="i am going to the office today"
)
print(f"BLEU Score: {bleu_score:.3f}")
```

## Performance Results

### BLEU Score Achievements
| Metric | Score | Status |
|--------|-------|----------|
| **Average BLEU** | **0.275** | Good Performance |
| **Peak BLEU** | **0.548** | Excellent |
| **Best Method** | **4-gram (k=0.005)** | Optimal |

### Top Performing Examples
| Input | Generated Completion | BLEU Score |
|-------|---------------------|------------|
| "this is" | "this is a 2-for-1 !" | **0.548** |
| "have a" | "have a great day !" | **0.387** |
| "i love" | "i love you !" | **0.318** |

### Model Comparison
| Model | BLEU Score | Context | Best Use Case |
|-------|------------|---------|---------------|
| **4-gram** | **0.369** | 3 words | Highest accuracy |
| **Trigram** | **0.327** | 2 words | Balanced performance |
| **Bigram** | **0.318** | 1 word | Fastest inference |

## Technical Details

### Model Architecture
- **N-gram Orders**: 1-gram through 4-gram models
- **Smoothing**: Additive (Laplace) smoothing with optimized parameters
- **Vocabulary**: 19,360+ unique tokens from social media data
- **Training**: 15,000 sentences with frequency-based filtering

### Optimization Features
- **Parameter Tuning**: Optimized smoothing (k=0.005)
- **Vocabulary Management**: Intelligent OOV handling with `<unk>` tokens
- **Boundary Handling**: Proper sentence start/end markers (`<s>`, `<e>`)
- **Memory Efficiency**: Scalable data structures for large vocabularies

### Training Data
- **Source**: English Twitter corpus (en_US.twitter.txt)
- **Size**: 47,961 sentences
- **Domain**: Contemporary social media language
- **Features**: Slang, abbreviations, informal expressions

## Research & Evaluation

### Academic Documentation
The project includes a comprehensive research paper (`research_paper.md`) covering:
- **Mathematical Foundations**: N-gram theory and BLEU methodology
- **Experimental Setup**: Detailed methodology and parameters
- **Results Analysis**: Quantitative and qualitative evaluation
- **Performance Comparison**: Multi-metric assessment
- **Future Work**: Extensions and improvements

### Evaluation Metrics
1. **BLEU Score**: Text generation quality assessment
2. **Perplexity**: Traditional language modeling metric
3. **Probability Rankings**: Prediction confidence analysis
4. **Contextual Relevance**: Qualitative completion assessment

## Development Features

### Interactive Components
- **Real-time Autocomplete**: Live word prediction as you type
- **Probability Display**: Confidence scores for each suggestion
- **Context Awareness**: Uses full input history for predictions
- **Graceful Handling**: Robust error handling and edge cases

### Text Generation
- **Markov Chains**: Bigram-based text generation
- **Social Media Style**: Twitter-appropriate output formatting
- **Customizable Length**: User-defined text generation length
- **Interactive Mode**: Real-time generation with user control

### Preprocessing Pipeline
- **Sentence Segmentation**: Intelligent text splitting
- **Tokenization**: NLTK-based word tokenization
- **Normalization**: Case conversion and punctuation handling
- **Vocabulary Construction**: Frequency-based word filtering

## Applications

### Real-World Use Cases
- **Mobile Keyboards**: Smartphone text prediction
- **Search Engines**: Query autocomplete functionality
- **Email Clients**: Smart compose features
- **Chat Applications**: Message completion assistance
- **Writing Tools**: Content suggestion systems

### Educational Applications
- **NLP Learning**: Understanding statistical language modeling
- **Research Baseline**: Benchmark for advanced models
- **Algorithm Study**: N-gram implementation patterns
- **Evaluation Methods**: BLEU score calculation techniques

## Advanced Usage

### Custom Configuration
```python
# Configure model parameters
model = NGramModel(
    max_n=4,                    # Maximum N-gram order
    count_threshold=1,          # Minimum word frequency
    smoothing_param=0.005       # Additive smoothing parameter
)

# Train with custom data
model.train(your_training_data, count_threshold=1)

# Get ranked predictions
predictions = model.get_suggestions(
    context="your input text",
    num_suggestions=10,
    n_order=4
)
```

### Custom Evaluation
```python
from eval_metrics import evaluate_autocomplete_bleu

# Run evaluation on your test set
results = evaluate_autocomplete_bleu()

# Custom BLEU calculation
from eval_metrics import calculate_bleu
bleu = calculate_bleu(
    candidate="generated text",
    reference="reference text"
)
```

## Contributing

We welcome contributions! Here's how you can help:

### Bug Reports
- Use GitHub Issues for bug reports
- Include steps to reproduce
- Provide sample inputs and expected outputs

### Feature Requests
- Suggest new evaluation metrics
- Propose model improvements
- Request additional documentation

### Development Setup
```bash
# Fork the repository
git clone https://github.com/your-username/beyond-autocomplete-ngram-llm.git

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python eval_metrics.py  # Ensure all tests pass

# Submit pull request
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NLTK Library**: Natural language processing toolkit
- **Twitter Dataset**: Social media training corpus
- **BLEU Methodology**: Text generation evaluation standard
- **Research Community**: Statistical language modeling foundations

## References

- Jurafsky, D. & Martin, J.H. "Speech and Language Processing"
- Chen, S.F. & Goodman, J. "An Empirical Study of Smoothing Techniques"
- Papineni, K. et al. "BLEU: A Method for Automatic Evaluation"

---

**Star this repository if you find it useful!**

**Contact**: [Your Email] | **GitHub**: [@arjungop](https://github.com/arjungop)

---

*Built with care for the NLP community*