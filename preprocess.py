# step 1: load data
# import necessary libraries
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import math
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

# load the original CSV file
df = pd.read_csv('malicious_phish.csv')

# step 2: handle missing values
# drop rows with missing values in 'url' or 'type' columns
df = df.dropna(subset=['url', 'type'])

# step 3: remove duplicate records
# remove duplicate URLs, keeping the first occurrence
df = df.drop_duplicates(subset=['url'], keep='first')

# step 4: feature extraction/encoding
# 4.1: url normalization
# function to normalize URLs: lowercase protocol/domain, drop fragments, strip default ports
def normalize_url(url):
    if pd.isna(url):
        return url
    
    url_str = str(url).strip()
    if not url_str:
        return url_str
    
    try:
        # add protocol if missing
        if not url_str.startswith(('http://', 'https://', 'http:', 'https:')):
            url_str = 'http://' + url_str
        
        # parse URL into components
        parsed = urlparse(url_str)
        
        # lowercase protocol and domain
        scheme = parsed.scheme.lower() if parsed.scheme else 'http'
        netloc = parsed.netloc.lower()
        
        # strip default ports (:80 for http, :443 for https)
        if netloc.endswith(':80') and scheme == 'http':
            netloc = netloc[:-3]
        elif netloc.endswith(':443') and scheme == 'https':
            netloc = netloc[:-4]
        
        # reconstruct URL without fragment (fragments are dropped automatically by not including parsed.fragment)
        path = parsed.path
        query = parsed.query
        
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        
        return normalized
    except (ValueError, Exception):
        # if URL parsing fails, return original URL (will be handled in feature extraction)
        return url_str

# apply normalization to all URLs
df['normalized_url'] = df['url'].apply(normalize_url)

# 4.2: extract lexical features
# function to calculate Shannon entropy (measures randomness/obfuscation in URL)
def calculate_entropy(text):
    if not text or len(text) == 0:
        return 0
    counter = Counter(text)
    length = len(text)
    entropy = 0
    for count in counter.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

# function to extract all 11 lexical features from a URL
def extract_features(url):
    if pd.isna(url):
        url = ''
    
    url_str = str(url)
    
    try:
        # parse URL (add protocol if needed)
        if not url_str.startswith(('http://', 'https://')):
            url_str = 'http://' + url_str
        parsed = urlparse(url_str)
        
        # extract URL components
        hostname = parsed.netloc.lower() if parsed.netloc else ''
        path = parsed.path if parsed.path else ''
        query = parsed.query if parsed.query else ''
    except (ValueError, Exception):
        # if parsing fails, use empty strings for components
        hostname = ''
        path = ''
        query = ''
    
    # length features (4 features)
    total_url_length = len(url_str)
    host_length = len(hostname)
    path_length = len(path)
    query_length = len(query)
    
    # count features (3 features)
    # count subdomains: number of dots minus 1 (e.g., www.example.com has 1 subdomain)
    subdomain_count = max(0, hostname.count('.') - 1) if hostname else 0
    
    # count digits in URL
    digit_count = sum(c.isdigit() for c in url_str)
    
    # count special symbols: -, _, @, ?, %, =, &, +
    special_symbols = ['-', '_', '@', '?', '%', '=', '&', '+']
    special_symbol_count = sum(url_str.count(sym) for sym in special_symbols)
    
    # count uppercase letters
    uppercase_count = sum(c.isupper() for c in url_str)
    
    # ratio features (3 features)
    digit_to_length_ratio = digit_count / total_url_length if total_url_length > 0 else 0
    symbol_to_length_ratio = special_symbol_count / total_url_length if total_url_length > 0 else 0
    uppercase_to_length_ratio = uppercase_count / total_url_length if total_url_length > 0 else 0
    
    # entropy feature (1 feature)
    shannon_entropy = calculate_entropy(url_str)
    
    # return all 11 features as a dictionary
    return {
        'total_url_length': total_url_length,
        'host_length': host_length,
        'path_length': path_length,
        'query_length': query_length,
        'subdomain_count': subdomain_count,
        'digit_count': digit_count,
        'special_symbol_count': special_symbol_count,
        'digit_to_length_ratio': digit_to_length_ratio,
        'symbol_to_length_ratio': symbol_to_length_ratio,
        'uppercase_to_length_ratio': uppercase_to_length_ratio,
        'shannon_entropy': shannon_entropy
    }

# extract features for all URLs and add to dataframe
features_df = df['normalized_url'].apply(lambda x: pd.Series(extract_features(x)))
df = pd.concat([df, features_df], axis=1)

# step 5: categorical encoding
# function to extract protocol type from URL
def extract_protocol(url):
    if pd.isna(url):
        return 'none'
    url_str = str(url)
    if url_str.startswith('https://'):
        return 'https'
    elif url_str.startswith('http://'):
        return 'http'
    else:
        return 'none'

# extract protocol and create one-hot encoded columns
df['protocol'] = df['normalized_url'].apply(extract_protocol)
df['has_http'] = (df['protocol'] == 'http').astype(int)
df['has_https'] = (df['protocol'] == 'https').astype(int)

# label encode target variable: benign=0, phishing=1, malware=2, defacement=3
label_encoder = LabelEncoder()
df['type_encoded'] = label_encoder.fit_transform(df['type'])

# step 6: feature scaling
# define list of numerical features to scale (all 11 lexical features)
numerical_features = [
    'total_url_length', 'host_length', 'path_length', 'query_length',
    'subdomain_count', 'digit_count', 'special_symbol_count',
    'digit_to_length_ratio', 'symbol_to_length_ratio', 'uppercase_to_length_ratio',
    'shannon_entropy'
]

# scale all numerical features using StandardScaler (mean=0, std=1)
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# step 7: handle imbalanced data
# prepare feature columns (11 numerical + 2 protocol columns = 13 features total)
feature_columns = numerical_features + ['has_http', 'has_https']
X = df[feature_columns]
y = df['type_encoded']

# apply random undersampling to balance all classes to match smallest class (malware)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# create balanced dataframe with all features and labels
df_balanced = pd.DataFrame(X_resampled, columns=feature_columns)
df_balanced['type_encoded'] = y_resampled
df_balanced['type'] = label_encoder.inverse_transform(y_resampled)

# step 8: save processed dataset
# save processed dataset to CSV file
df_balanced.to_csv('processed_data.csv', index=False)
