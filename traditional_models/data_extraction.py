"""
Cricket Data Extraction Module
Extracts numerical features from aggregated cricket stats and player command files
"""

import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import os
from collections import defaultdict

class CricketDataExtractor:
    def __init__(self, base_path="."):
        """Initialize paths and player information"""
        self.base_path = Path(base_path)
        self.llm_path = self.base_path / ".." / "llm"
        self.aggregated_path = self.base_path / ".." / "cricinfo_data_scrapping" / "aggregated_stats"
        self.against_teams_path = self.base_path / ".." / "cricinfo_data_scrapping" / "aggregated_stats_against_teams"
        self.venue_wise_path = self.base_path / ".." / "cricinfo_data_scrapping" / "aggregated_stats_venue_wise"
        
        # Player list with expected match counts
        self.players = {
            "Shubman Gill": 55,
            "Babar Azam": 125, 
            "Daryl Mitchell": 44,
            "Harry Tector": 46,
            "Charith Asalanka": 67,
            "Ibrahim Zadran": 36,
            "Shai Hope": 128
        }
        
        # Dismissal types for label encoding
        self.dismissal_types = [
            'Caught-Fielder', 'Bowled', 'LBW', 'Caught-Keeper', 
            'Run-Out', 'Stumped', 'Hit-Wicket', 'Not-Out'
        ]
        
    def load_aggregated_stats(self, player_name):
        """Load all aggregated stats for a player"""
        # Clean player name for filename
        clean_name = player_name.lower().replace(' ', '_')
        
        stats = {}
        
        # Load general stats
        general_file = self.aggregated_path / f"{clean_name}_all_span_stats.json"
        if general_file.exists():
            with open(general_file, 'r') as f:
                stats['general'] = json.load(f)
        
        # Load against teams stats
        against_teams_file = self.against_teams_path / f"{clean_name}_against_teams_all_span_stats.json"
        if against_teams_file.exists():
            with open(against_teams_file, 'r') as f:
                stats['against_teams'] = json.load(f)
        
        # Load venue-wise stats
        venue_file = self.venue_wise_path / f"{clean_name}_venue_wise_all_span_stats.json"
        if venue_file.exists():
            with open(venue_file, 'r') as f:
                stats['venue_wise'] = json.load(f)
        
        return stats
    
    def parse_command_line(self, command_line):
        """Parse a single command line to extract match scenario details"""
        # Extract quoted parameters
        pattern_quoted = r'--(\w+)\s+"([^"]*)"'
        matches_quoted = re.findall(pattern_quoted, command_line)
        
        # Extract non-quoted parameters  
        pattern_unquoted = r'--(\w+)\s+(\S+)'
        matches_unquoted = re.findall(pattern_unquoted, command_line)
        
        params = {}
        
        # Add quoted parameters first (higher priority)
        for key, value in matches_quoted:
            params[key] = value
            
        # Add non-quoted parameters if not already present
        for key, value in matches_unquoted:
            if key not in params:
                params[key] = value
        
        return params
    
    def load_player_scenarios(self, player_name):
        """Load match scenarios from player command file"""
        # Clean player name for filename
        clean_name = player_name.lower().replace(' ', '_')
        command_file = self.llm_path / f"{clean_name}_commands.txt"
        
        scenarios = []
        
        if not command_file.exists():
            print(f"Warning: Command file not found for {player_name}: {command_file}")
            return scenarios
        
        with open(command_file, 'r') as f:
            lines = f.readlines()
        
        expected_scenarios = self.players[player_name]
        
        for i, line in enumerate(lines):
            if i >= expected_scenarios:
                break
                
            line = line.strip()
            if line and (line.startswith('python predict.py') or line.startswith('python driver.py')):
                params = self.parse_command_line(line)
                if params:
                    scenarios.append(params)
        
        print(f"Loaded {len(scenarios)} scenarios for {player_name} (expected: {expected_scenarios})")
        return scenarios
    
    def extract_numerical_features_from_stats(self, stats_dict):
        """Extract numerical features from aggregated stats"""
        features = {}
        
        # Extract features from each stats category
        for category, stats in stats_dict.items():
            if isinstance(stats, dict):
                # Recursively extract from nested dictionaries
                if any(isinstance(v, dict) for v in stats.values()):
                    for key, value in stats.items():
                        if isinstance(value, dict):
                            nested_features = self.extract_numerical_features_from_stats({key: value})
                            for nested_key, nested_value in nested_features.items():
                                features[f"{category}_{nested_key}"] = nested_value
                        elif isinstance(value, (int, float)):
                            features[f"{category}_{key}"] = value
                else:
                    # Direct numerical values
                    for key, value in stats.items():
                        if isinstance(value, (int, float)):
                            features[f"{category}_{key}"] = value
        
        return features
    
    def get_contextual_features(self, scenario, player_stats):
        """Extract contextual features based on scenario details"""
        features = {}
        
        # Get general stats features
        if 'general' in player_stats:
            general_features = self.extract_numerical_features_from_stats(player_stats['general'])
            for key, value in general_features.items():
                features[f"general_{key}"] = value
        
        # Get opponent-specific features if available
        opponent = scenario.get('opponent_team', '')
        if 'against_teams' in player_stats and opponent:
            against_teams = player_stats['against_teams']
            # Look for opponent in against_teams stats
            for team_data in against_teams.values():
                if isinstance(team_data, dict) and opponent.lower() in str(team_data).lower():
                    team_features = self.extract_numerical_features_from_stats(team_data)
                    for key, value in team_features.items():
                        features[f"vs_team_{key}"] = value
                    break
        
        # Get venue-specific features if available
        venue = scenario.get('venue', '')
        if 'venue_wise' in player_stats and venue:
            venue_wise = player_stats['venue_wise']
            # Look for venue in venue_wise stats
            for venue_data in venue_wise.values():
                if isinstance(venue_data, dict) and any(venue.lower() in str(v).lower() for v in venue_data.values() if isinstance(v, str)):
                    venue_features = self.extract_numerical_features_from_stats(venue_data)
                    for key, value in venue_features.items():
                        features[f"venue_{key}"] = value
                    break
        
        return features
    
    def create_feature_matrix(self):
        """Create feature matrix and metadata for all players and scenarios"""
        all_features = []
        all_metadata = []
        
        print("Starting feature extraction...")
        print("="*50)
        
        for player_name in self.players.keys():
            print(f"\nProcessing {player_name}...")
            
            # Load player stats
            player_stats = self.load_aggregated_stats(player_name)
            print(f"  Loaded stats categories: {list(player_stats.keys())}")
            
            # Load player scenarios
            scenarios = self.load_player_scenarios(player_name)
            
            for i, scenario in enumerate(scenarios):
                # Extract contextual features
                features = self.get_contextual_features(scenario, player_stats)
                
                # Create metadata
                metadata = {
                    'player': player_name,
                    'scenario_id': i,
                    'match_date': scenario.get('match_date', ''),
                    'opponent_team': scenario.get('opponent_team', ''),
                    'venue': scenario.get('venue', ''),
                    'ground_truth': scenario.get('ground_truth', 'Unknown')
                }
                
                all_features.append(features)
                all_metadata.append(metadata)
        
        # Convert to DataFrames
        features_df = pd.DataFrame(all_features)
        metadata_df = pd.DataFrame(all_metadata)
        
        # Fill NaN values with 0 (missing contextual information)
        features_df = features_df.fillna(0)
        
        print(f"\nFeature extraction completed!")
        print(f"Total scenarios: {len(features_df)}")
        print(f"Total features: {features_df.shape[1]}")
        print(f"Feature types: {features_df.dtypes.value_counts().to_dict()}")
        
        # Remove scenarios with unknown ground truth
        valid_mask = metadata_df['ground_truth'] != 'Unknown'
        features_df = features_df[valid_mask]
        metadata_df = metadata_df[valid_mask]
        
        print(f"After removing unknown ground truth: {len(features_df)} scenarios")
        
        return features_df, metadata_df
    
    def save_extracted_data(self, features_df, metadata_df):
        """Save extracted features and metadata"""
        output_dir = Path("processed_data")
        output_dir.mkdir(exist_ok=True)
        
        # Save features and metadata
        features_file = output_dir / "features.csv"
        metadata_file = output_dir / "metadata.csv"
        
        features_df.to_csv(features_file, index=False)
        metadata_df.to_csv(metadata_file, index=False)
        
        print(f"\nData saved:")
        print(f"  Features: {features_file}")
        print(f"  Metadata: {metadata_file}")
        
        # Generate summary statistics (convert numpy types to native Python types)
        summary = {
            'extraction_timestamp': datetime.now().isoformat(),
            'total_scenarios': int(len(features_df)),
            'total_features': int(features_df.shape[1]),
            'players_processed': list(self.players.keys()),
            'dismissal_distribution': {k: int(v) for k, v in metadata_df['ground_truth'].value_counts().to_dict().items()},
            'feature_summary': {
                'numerical_features': int(features_df.shape[1]),
                'missing_values': int(features_df.isnull().sum().sum()),
                'zero_values': int((features_df == 0).sum().sum())
            }
        }
        
        summary_file = output_dir / "extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  Summary: {summary_file}")
        
        return features_file, metadata_file, summary_file

def main():
    """Main execution function"""
    print("Cricket Data Extraction Module")
    print("="*50)
    
    # Initialize extractor
    extractor = CricketDataExtractor()
    
    # Extract features
    features_df, metadata_df = extractor.create_feature_matrix()
    
    # Save data
    extractor.save_extracted_data(features_df, metadata_df)
    
    print("\nData extraction completed successfully!")

if __name__ == "__main__":
    main() 