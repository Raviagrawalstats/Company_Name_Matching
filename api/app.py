import streamlit as st
import pandas as pd
import numpy as np
import warnings
import re
import string
from name_matching.name_matcher import NameMatcher

warnings.filterwarnings('ignore')

st.title('Company Name Matcher')

# File upload widgets
st.subheader('Upload CSV Files')
csv_file1 = st.file_uploader("Upload first CSV file", type=['csv'])
csv_file2 = st.file_uploader("Upload second CSV file", type=['csv'])

# Parameters
col1, col2 = st.columns(2)
with col1:
    num_matches = st.slider('Number of matches', 2, 50, 5)
with col2:
    threshold = st.slider('Threshold', 10, 100, 95)

# Add submit button
submit_button = st.button('Process Company Names')

if submit_button and csv_file1 is not None and csv_file2 is not None:
    with st.spinner('Processing... Please wait.'):
        # Read CSV files
        df1 = pd.read_csv(csv_file1)
        df2 = pd.read_csv(csv_file2)

        company_names = df1.dropna()
        match_names = df2.dropna()

        # Remove integer values from company names
        comp_names = [name for name in company_names.iloc[:, 0] if isinstance(name, str)]
        match_comp_names = [name for name in match_names.iloc[:, 0] if isinstance(name, str)]

        word_removal_pattern = r"\b(?:ltd\.?|pvt\.?|m/s|the|m/s.|private|llp|pvt|company|ltd|limited|services|service|limi|limite|independent landlord|independent developer)\b"
        
        # Clean company names
        def clean_names(names):
            cleaned = []
            for name in names:
                name = name.lower()
                name = name.translate(str.maketrans("", "", string.punctuation))
                name = re.sub(word_removal_pattern, "", name)
                name = re.sub(r"[^\w\s]", "", name)
                cleaned.append(name)
            return cleaned

        clean_company_names = clean_names(comp_names)
        clean_match_names = clean_names(match_comp_names)
        
        comp_names = pd.DataFrame(clean_company_names, columns=['company_name'])
        match_comp_names = pd.DataFrame(clean_match_names, columns=['company_name'])

        comp_names[['First Word', 'Rest']] = comp_names['company_name'].str.split(n=1, expand=True)
        match_comp_names[['First Word', 'Rest']] = match_comp_names['company_name'].str.split(n=1, expand=True)

        # First word matching
        matcher = NameMatcher(
            number_of_matches=num_matches,
            lowercase=True,
            punctuations=True,
            remove_ascii=True,
            legal_suffixes=True, 
            common_words=False, 
            top_n=50, 
            verbose=True
        )

        matcher.set_distance_metrics([
            'discounted_levenshtein',
            'weighted_jaccard',
            'fuzzy_wuzzy_token_sort'
        ])

        matcher.load_and_process_master_data(
            column='First Word',
            df_matching_data=match_comp_names, 
            transform=True
        )

        matches_2 = matcher.match_names(
            to_be_matched=comp_names, 
            column_matching='First Word'
        )
        
        # Full name matching
        matcher = NameMatcher(
            number_of_matches=num_matches,
            lowercase=True,
            punctuations=True,
            remove_ascii=True,
            legal_suffixes=True, 
            common_words=False, 
            top_n=50, 
            verbose=True
        )

        matcher.set_distance_metrics([
            'discounted_levenshtein',
            'weighted_jaccard',
            'fuzzy_wuzzy_token_sort'
        ])

        matcher.load_and_process_master_data(
            column='company_name',
            df_matching_data=match_comp_names, 
            transform=True
        )

        matches_1 = matcher.match_names(
            to_be_matched=comp_names, 
            column_matching='company_name'
        )
        
        matches = matches_2.join(matches_1, lsuffix='_left', rsuffix='_right')

        # Process matches
        match_index_columns = [col for col in matches.columns if col.startswith('match_index')]

        for index, row in matches.iterrows():
            for col in match_index_columns:
                score_col = 'score_' + col[col.rindex('_')-1] + col[col.rindex('_'):]
                match_index_value = row[col]
                
                if match_index_value == index:
                    matches.at[index, score_col] = 0

        score_right_cols = matches.filter(regex=r'^score.*right$', axis=1)
        score_left_cols = matches.filter(regex=r'^score.*left$', axis=1)

        score_left_mask = score_left_cols.eq(100).any(axis=1)
        score_right_mask = score_right_cols.gt(threshold).any(axis=1)

        filtered_df_any = matches[score_right_mask & score_left_mask]

        col_filter = (num_matches*3)+1

        values = filtered_df_any.iloc[:,col_filter:].values.tolist()
        column_names = filtered_df_any.iloc[:,col_filter:].columns.tolist()

        result_df = pd.DataFrame(values, columns=column_names)

        result_df["id"] = result_df.index
        Output = pd.wide_to_long(
            result_df,
            stubnames=['match_name', 'score', 'match_index'],
            i=['id','original_name_right'],
            j='match_number',
            sep='_',
            suffix='.+'
        )
        Output.reset_index(inplace=True)
        Output = Output.drop(columns='match_number', axis=1)
        Output = Output.rename(columns={'original_name_right': 'original_name'})

        Output = Output.applymap(lambda x: int(x) if isinstance(x, np.int64) else x)

        Output_filtered = Output[Output["score"] != 0]

        # Display results
        st.subheader('Results')
        st.dataframe(Output_filtered)

        # Download button
        csv = Output_filtered.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="matched_companies.csv",
            mime="text/csv"
        )
elif submit_button and (csv_file1 is None or csv_file2 is None):
    st.error('Please upload both CSV files before processing.')