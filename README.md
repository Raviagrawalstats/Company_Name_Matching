---
title: Company Name Matcher
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# Company Name Matcher 🏢

A powerful Streamlit application for matching company names using advanced string matching algorithms.

## Features

- **Smart Name Matching**: Uses multiple algorithms including discounted Levenshtein distance, weighted Jaccard similarity, and fuzzy string matching
- **Data Cleaning**: Automatically removes common legal suffixes (Ltd, Inc, LLC, etc.) and punctuation for better matching
- **Customizable Parameters**: Adjust the number of matches and similarity threshold
- **Easy CSV Upload**: Simply upload two CSV files with company names
- **Downloadable Results**: Export matched results as CSV

## How to Use

1. **Upload CSV Files**: Upload two CSV files containing company names in the first column
   - First file: Companies you want to match
   - Second file: Reference companies to match against

2. **Set Parameters**:
   - **Number of matches**: How many potential matches to show per company (2-50)
   - **Threshold**: Minimum similarity score percentage (10-100%)

3. **Process**: Click "Process Company Names" to start matching

4. **Download**: Download the results as a CSV file

## Input Format

Your CSV files should have company names in the first column. The app will automatically:
- Remove common legal suffixes (Ltd, Inc, LLC, Private, Limited, etc.)
- Clean punctuation and special characters
- Normalize case for better matching

## Matching Algorithm

The app uses a sophisticated two-stage matching process:
1. **First Word Matching**: Matches based on the first word of company names
2. **Full Name Matching**: Matches based on complete company names

Uses multiple distance metrics:
- Discounted Levenshtein Distance
- Weighted Jaccard Similarity
- Fuzzy Wuzzy Token Sort

## Example

Input companies:
- "Microsoft Corporation"
- "Apple Inc."
- "Google LLC"

Reference companies:
- "Microsoft Corp"
- "Apple Incorporated"
- "Alphabet Inc"

The app will find high-confidence matches between similar company names.

---

Built with ❤️ using Streamlit and the name-matching library.
