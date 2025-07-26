# Context is Everything: Boosting Cricket Dismissal Prediction with Large Language Models

## Abstract

Cricket ranks as the world's second-most popular sport with approximately 2.5 billion fans, yet predicting how a batsman will be dismissed remains exceptionally challenging. Each dismissal type (bowled, caught, LBW, stumped, run‑out, etc.) is shaped by layered contextual factors such as venue, pitch, weather, opponent, toss, bowler style, field setup, and match state. Traditional predictive models, including Bayesian survival analysis, hidden Markov models, and random forests, often overlook critical context due to data sparsity, high-dimensionality, and operational constraints, leading to limited scalability and frequent retraining needs. While Large Language Models (LLMs) offer the ability to tailor predictions to each player through in-context learning, it remains unknown which contextual inputs maximize prediction accuracy.

In this research, we systematically evaluate LLMs for batsman dismissal prediction by evaluating the impact of different contextual inputs. Through comprehensive ablation experiments using structured (career statistics, venue and opponent history) and unstructured (recent match commentary) contextual inputs across multiple LLM variants, we find that including venue and opponent context elevates top-1 accuracy from 43% (career stats alone) to over 56%, and top-2 accuracy from 57% to nearly 80%. Among tested models, GPT-4.1-mini offers the optimal trade-off between accuracy and efficiency for practical deployment.

## 🎯 Research Contributions

This repository contains the complete implementation and experimental framework for our research on LLM-based cricket dismissal prediction. Our key contributions include:

1. **Comprehensive Ablation Study**: Systematic evaluation of contextual input impact on prediction accuracy
2. **Multi-Model Comparison**: Analysis across GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, and o4-mini variants
3. **Contextual Input Framework**: Novel approach combining structured statistics with unstructured commentary
4. **Scalable Prediction System**: Practical framework for real-world cricket analytics deployment
5. **Extensive Dataset**: Large-scale cricket data collection and processing pipeline

## 📊 Experimental Results

### Key Findings
- **Venue + Team Context**: Improves top-1 accuracy from 43% (career stats) to 57.5% (o4-mini)
- **Top-2 Accuracy**: Enhanced from 58% to 79.2% with optimal contextual inputs
- **Best Performance**: o4-mini achieves highest accuracy (57.5% top-1, 79.2% top-2)
- **Contextual Impact**: Venue + Team statistics significantly outperform commentary-based inputs

### Model Performance Comparison
| Model | Top-1 Accuracy | Top-2 Accuracy | Best Configuration |
|-------|----------------|----------------|-------------------|
| GPT-4.1-nano | 51.3% | 68.5% | Venue + Team Stats |
| GPT-4.1 | 55.3% | 77.0% | Venue + Team Stats |
| GPT-4.1-mini | 56.3% | 76.8% | Venue + Team Stats |
| o4-mini | 57.5% | 79.2% | Venue + Team Stats |

*Results based on optimal contextual input combination (Venue Stats + Vs-Team Stats) across 501 test cases*

## 🏗️ Repository Structure

```
SPROJ-PCB/
├── cricinfo_data_scrapping/          # Data collection and preprocessing
│   ├── Scripts/                      # Web scraping and data processing
│   ├── aggregated_stats/             # Player statistics by date
│   ├── aggregated_stats_against_teams/ # Team-wise performance data
│   ├── aggregated_stats_venue_wise/  # Venue-specific statistics
│   ├── CommentaryData/               # Ball-by-ball commentary
│   └── match_scorecards_json/        # Match scorecard data
├── llm/                              # Core research implementation
│   ├── driver.py                     # Main experimental framework
│   ├── automate_run.py               # Automated ablation experiments
│   ├── plots/                        # Result visualizations
│   ├── filtered_outputs/             # Processed contextual inputs
│   └── player_matches/               # Player-specific datasets
├── dbcode/                           # Database schema and management
├── ipynb_analysis/                   # Exploratory data analysis
└── Old_Data/                         # Legacy datasets and utilities
```

## 🔬 Research Methodology

### 1. Data Collection Pipeline
Our comprehensive data collection framework extracts:
- **Match Scorecards**: Detailed batting/bowling statistics
- **Ball-by-Ball Commentary**: Unstructured match descriptions
- **Player Statistics**: Career performance metrics
- **Venue Information**: Ground-specific data
- **Opposition Analysis**: Team-wise performance patterns

### 2. Contextual Input Framework
We evaluate multiple contextual input combinations:

#### Structured Contexts
- **Career Statistics**: Overall batting performance metrics
- **Venue History**: Performance at specific cricket grounds
- **Opponent Analysis**: Historical performance against teams
- **Recent Form**: Recent match performance trends

#### Unstructured Contexts
- **Match Commentary**: Recent ball-by-ball descriptions
- **Match State**: Current game situation and pressure
- **Field Settings**: Fielding positions and strategies

### 3. Ablation Study Design
Systematic evaluation of contextual input impact:
1. **Baseline**: Career statistics only
2. **Venue Context**: + venue-specific performance
3. **Opponent Context**: + team-wise performance
4. **Commentary Context**: + recent match descriptions
5. **Combined Contexts**: All contextual inputs

### 4. Model Evaluation Framework
- **Multi-Model Testing**: GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, o4-mini
- **Accuracy Metrics**: Top-1 and Top-2 prediction accuracy
- **Efficiency Analysis**: Computational cost vs. performance trade-offs
- **Statistical Significance**: Confidence intervals and hypothesis testing

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PostgreSQL (for database functionality)
Chrome/Chromium (for web scraping)
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd SPROJ-PCB

# Install dependencies
pip install -r requirements.txt

# Setup database (optional)
cd dbcode
python createdb.py
```

### Running Experiments

#### 1. Data Collection
```bash
cd cricinfo_data_scrapping/Scripts

# Complete data pipeline
python get_odi_series_urls.py
python get_scorecard_commentary_urls.py
python get_scorecard.py
python add_btype_scorecards.py
python add_batting_type_scorecards.py
python add_playingxi.py
python get_player_stats.py
python get_player_stats_venuewise.py
python get_player_stats_against_teams.py
```

#### 2. LLM Experiments
```bash
cd llm

# Run complete ablation study
python automate_run.py

# Generate result visualizations
python plot_acc.py
python plot_aggregate_by_model.py
python plot_aggregate_by_player.py
```

#### 3. Individual Player Analysis
```bash
# Analyze specific player (e.g., Babar Azam)
python driver.py --player "Babar Azam" --model gpt-4.1-mini
```

## 📈 Results and Visualizations

The research generates comprehensive outputs:

### Accuracy Analysis
- **Top-1/Top-2 Accuracy Plots**: Model performance comparison
- **Confidence Intervals**: Statistical significance analysis
- **Recall Heatmaps**: Dismissal type prediction patterns

### Ablation Study Results
- **Context Impact Analysis**: Effect of different inputs
- **Model Efficiency Comparison**: Performance vs. computational cost
- **Player-Specific Analysis**: Individual player prediction patterns

### Statistical Reports
- **Aggregated Results**: CSV files with detailed metrics
- **Model Comparison Tables**: Side-by-side performance analysis
- **Significance Testing**: Statistical validation of results

## 🔍 Key Scripts

### Research Framework
- `llm/driver.py`: Main experimental framework
- `llm/automate_run.py`: Automated ablation experiments
- `llm/plot_acc.py`: Accuracy analysis and visualization

### Data Processing
- `cricinfo_data_scrapping/Scripts/get_player_stats.py`: Player statistics generation
- `cricinfo_data_scrapping/Scripts/get_player_stats_venuewise.py`: Venue-specific analysis
- `cricinfo_data_scrapping/Scripts/filter_commentary.py`: Commentary processing

### Analysis Tools
- `llm/plot_aggregate_by_model.py`: Model comparison analysis
- `llm/plot_aggregate_by_player.py`: Player-specific analysis
- `ipynb_analysis/Player_Dismissals.ipynb`: Exploratory analysis

## 📊 Dataset Information

### Player Coverage
- **Pakistan**: Babar Azam, Azhar Ali, Fakhar Zaman, Shoaib Malik, Mohammad Hafeez
- **India**: Virat Kohli, Rohit Sharma, Shikhar Dhawan, MS Dhoni
- **Australia**: Steven Smith
- **Other Teams**: Charith Asalanka (SL), Daryl Mitchell (NZ), Harry Tector (IRE), Ibrahim Zadran (AFG), Shai Hope (WI), Shubman Gill (IND)

### Data Volume
- **Matches**: 3,000+ ODI matches
- **Commentary**: 1,700+ processed commentary files
- **Statistics**: 600+ aggregated player datasets
- **Venues**: 100+ cricket grounds analyzed

## 🎓 Research Impact

This work establishes:
1. **Optimal Contextual Inputs**: Clear guidance on which factors enhance prediction accuracy
2. **Scalable Framework**: Practical system for real-world cricket analytics
3. **Model Selection Guidelines**: Trade-off analysis for deployment decisions
4. **Future Research Directions**: Systematic approach for sports prediction

## 📚 Citation

If you use this research in your work, please cite our paper:

```bibtex
@inproceedings{cricket-dismissal-prediction-2026,
  title={Context is Everything: Boosting Cricket Dismissal Prediction with Large Language Models},
  author={Mustafa Ashfaq and Essa Shahid Arshad},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```

## 👥 Authors

- **Mustafa Ashfaq** - [GitHub Profile](https://github.com/Mustafa-Ashfaq81)
- **Essa Shahid Arshad** - [GitHub Profile](https://github.com/essashahid)

## 🤝 Contributing

We welcome contributions to improve the research framework:
- Bug reports and feature requests
- Additional ablation experiments
- Enhanced visualization tools
- Extended dataset coverage

## 📄 License

This research is released under the MIT License. Please respect ESPNcricinfo's terms of service when using the web scraping components.

## 🔗 Related Work

- Cricket Analytics and Performance Prediction
- Large Language Models in Sports Analytics
- Contextual Learning in Predictive Modeling
- Ablation Studies in Machine Learning

---

**Research Note**: This repository contains the complete experimental framework for our research on LLM-based cricket dismissal prediction. All results, visualizations, and analysis tools are provided for reproducibility and further research.