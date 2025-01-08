import os
import re
import spacy

# Update file paths for local system
COMMENTARY_FILE_PATH = "../CommentaryData/raw_commentary/India_vs_Australia_WorldCup_Chennai_2023.txt"
PLAYER_NAME = "Virat Kohli"
ALTERNATE_NAME = "Kohli"
OUTPUT_DIRECTORY = "../CommentaryData/filtered_commentary"

# Generate dynamic output file name and path
OUTPUT_FILE_NAME = f"{PLAYER_NAME.replace(' ', '_').lower()}_filtered_comm.txt"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE_NAME)

PRONOUN_PATTERN = re.compile(r"\b(he|his|him)\b", re.IGNORECASE)
CONTEXT_WINDOW = 2


def read_commentary(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Commentary file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def extract_player_sentences(text: str, nlp, player_name: str, alt_name: str) -> list:
    pn_lower = player_name.lower()
    alt_lower = alt_name.lower()
    doc = nlp(text)
    extracted = []
    player_context_counter = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_lower = sent_text.lower()
        direct_mention = (pn_lower in sent_lower) or (alt_lower in sent_lower)
        mentions_pronoun = bool(PRONOUN_PATTERN.search(sent_text))
        keep_sentence = False

        if direct_mention:
            keep_sentence = True
            player_context_counter = CONTEXT_WINDOW
        else:
            if player_context_counter > 0 and mentions_pronoun:
                keep_sentence = True
                player_context_counter -= 1
            else:
                if player_context_counter > 0:
                    player_context_counter -= 1

        if keep_sentence:
            extracted.append(sent_text)

    return extracted


def clean_extracted_sentences(extracted_sentences: list) -> list:
    seen = set()
    cleaned = []
    for sentence in extracted_sentences:
        sentence_stripped = sentence.strip()
        if sentence_stripped and sentence_stripped not in seen:
            cleaned.append(sentence_stripped)
            seen.add(sentence_stripped)
    return cleaned


def validate_output(raw_text: str, filtered_sentences: list, player_name: str):
    raw_words = raw_text.split()
    total_raw_words = len(raw_words)
    extracted_text = " ".join(filtered_sentences)
    extracted_words = extracted_text.split()
    total_extracted_words = len(extracted_words)

    if total_raw_words > 0:
        extraction_percentage = (total_extracted_words / total_raw_words) * 100
    else:
        extraction_percentage = 0.0

    player_name_lower = player_name.lower()
    player_mentions_raw = raw_text.lower().count(player_name_lower)
    player_mentions_extracted = extracted_text.lower().count(player_name_lower)

    print(f"Total Words in Raw Commentary: {total_raw_words}")
    print(f"Total Words in Extracted Commentary: {total_extracted_words}")
    print(f"Percentage of Extracted Words: {extraction_percentage:.2f}%")
    print(f"Player Name Occurrences in Raw Commentary: {player_mentions_raw}")
    print(f"Player Name Occurrences in Extracted Commentary: {player_mentions_extracted}")

    if player_mentions_extracted < player_mentions_raw:
        print(f"Some mentions of '{player_name}' may have been missed.")
    elif player_mentions_extracted > player_mentions_raw:
        print(f"More mentions of '{player_name}' found than expected. Possible duplicates or overlapping references?")
    else:
        print(f"All mentions of '{player_name}' in raw text appear to be captured in the filtered text.")


def main():
    print("Loading spaCy English model...")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded.")

    print(f"Reading commentary from {COMMENTARY_FILE_PATH}...")
    raw_commentary_text = read_commentary(COMMENTARY_FILE_PATH)
    print("Commentary loaded.")
    print("-"*100)

    print(f"Extracting sentences specifically about '{PLAYER_NAME}' (and pronouns referencing them)...")
    extracted = extract_player_sentences(
        raw_commentary_text,
        nlp,
        player_name=PLAYER_NAME,
        alt_name=ALTERNATE_NAME
    )
    print(f"Total extracted sentences (initial): {len(extracted)}")
    print("-"*100)

    print("Cleaning extracted sentences...")
    cleaned_sentences = clean_extracted_sentences(extracted)

    print(f"Writing final results to {OUTPUT_FILE_PATH}...")
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as outfile:
        for sentence in cleaned_sentences:
            outfile.write(sentence + "\n")
    print("-"*100)

    validate_output(raw_commentary_text, cleaned_sentences, PLAYER_NAME)
    print("-"*100)

    print("Processing complete. Check the output file for results.")


if __name__ == "__main__":
    main()
