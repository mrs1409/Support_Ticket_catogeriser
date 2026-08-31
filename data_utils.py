import pandas as pd
import re
import string

# Common abbreviation expansions typed by real users
_ABBREV = {
    r"\bpls\b": "please",
    r"\bplz\b": "please",
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\bim\b": "i am",
    r"\bi'm\b": "i am",
    r"\bcant\b": "cannot",
    r"\bwont\b": "will not",
    r"\bdont\b": "do not",
    r"\bdoesnt\b": "does not",
    r"\bdidnt\b": "did not",
    r"\bhasnt\b": "has not",
    r"\bhavent\b": "have not",
    r"\bisnt\b": "is not",
    r"\barent\b": "are not",
    r"\bwasnt\b": "was not",
    r"\bwerent\b": "were not",
    r"\bcouldnt\b": "could not",
    r"\bshouldnt\b": "should not",
    r"\bwouldnt\b": "would not",
    r"\bthx\b": "thanks",
    r"\bthnx\b": "thanks",
    r"\bty\b": "thank you",
    r"\bbtw\b": "by the way",
    r"\basap\b": "as soon as possible urgent",
    r"\bomg\b": "urgent",
    r"\bwtf\b": "very frustrated urgent",
    r"\bacc\b": "account",
    r"\bsubscr\b": "subscription",
    r"\bpmt\b": "payment",
    r"\binv\b": "invoice",
    r"\bdbl\b": "double",
    r"\bdup\b": "duplicate",
    r"\bpwd\b": "password",
    r"\bpw\b": "password",
    r"\b2fa\b": "two factor authentication",
    r"\bapp\b": "application",
    r"\bcrash(ing|ed)?\b": "crashing crash",
    r"\bglitch(ing|ed)?\b": "glitch error",
    r"\bfroze\b": "frozen not responding",
    r"\bfrozen\b": "frozen not responding",
    r"\bhang(ing|s)?\b": "hanging not responding",
    r"\bbugged\b": "bug error",
    r"\bbroken\b": "broken not working",
    r"\bperms\b": "permissions",
    r"\bppl\b": "people",
    r"\brn\b": "right now",
    r"\bdunno\b": "do not know",
    r"\b!{2,}\b": "",
    r"\?{2,}": "?",
}

def normalize_input(text: str) -> str:
    """
    Normalize casually-typed user input to match training data style.
    Expands abbreviations, fixes common informal patterns.
    Apply this to ALL text before clean_text() at both train and predict time.
    """
    t = str(text).lower().strip()
    # Remove repeated punctuation
    t = re.sub(r'[!]{2,}', ' urgent ', t)
    t = re.sub(r'[?]{2,}', '? ', t)
    t = re.sub(r'\.{3,}', ' ', t)
    # Expand abbreviations
    for pattern, replacement in _ABBREV.items():
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    # Remove excessive spaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Raw data: {df.shape[0]} rows, columns: {df.columns.tolist()}")

    # Find the label column (handles different dataset formats)
    label_col = None
    for c in ['Ticket Type', 'ticket_type', 'type', 'category', 'Category', 'label', 'Topic_group']:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {df.columns.tolist()}")

    # Find text columns
    desc_col = None
    for c in ['Ticket Description', 'description', 'text', 'body', 'Ticket_Description', 'Document']:
        if c in df.columns:
            desc_col = c
            break

    subj_col = None
    for c in ['Ticket Subject', 'subject', 'title', 'Subject', 'Ticket_Subject']:
        if c in df.columns:
            subj_col = c
            break

    if desc_col is None:
        raise ValueError(f"No description column. Columns: {df.columns.tolist()}")

    # Combine subject + description
    if subj_col:
        df['text'] = df[subj_col].fillna('') + ' ' + df[desc_col].fillna('')
    else:
        df['text'] = df[desc_col].fillna('')

    # normalize_input() closes the style gap between formal training text and
    # casually-typed real user input (abbreviations, repeated punctuation) —
    # applied here at train time and again in predict.py at inference time.
    df['text_clean'] = df['text'].apply(lambda x: clean_text(normalize_input(x)))
    df['label'] = df[label_col].astype(str).str.strip()

    # Drop rows with empty text or unknown labels
    df = df[df['text_clean'].str.len() > 5]
    df = df[df['label'].notna()]
    df = df[df['label'] != 'nan']

    # Remove extremely rare labels
    counts = df['label'].value_counts()
    valid = counts[counts >= 10].index
    df = df[df['label'].isin(valid)]

    print(f"Clean data: {df.shape[0]} rows, {df['label'].nunique()} categories")
    print(df['label'].value_counts().to_string())
    return df[['text', 'text_clean', 'label']].reset_index(drop=True)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_label_distribution(df: pd.DataFrame) -> pd.Series:
    return df['label'].value_counts()

# Naturally-typed examples mirroring real user writing style, mapped to this
# project's actual 8 IT-service-desk categories (Access, Hardware, HR Support,
# Storage, Purchase, Administrative rights, Internal Project, Miscellaneous —
# the real Kaggle taxonomy, not the earlier abandoned synthetic 5-category set).
_NATURAL_EXAMPLES = {
    'Access': [
        "cant login my acc is locked out pls help",
        "forgot my pwd need reset asap",
        "locked out again wtf is going on",
        "cant get into shared drive perms are wrong",
        "2fa code never came thru pls fix",
        "keep getting invalid login error",
        "need vpn access for remote work asap",
        "acc got disabled dunno why",
        "cant reset my password link is broken",
        "denied access to the shared folder again",
        # Edge-case fixes (was misclassified as Miscellaneous)
        "my login doesnt work at all plz fix",
        "cant log into my account please help asap",
        "login is broken wont let me in at all",
        "sign in page wont work keeps rejecting my password",
        "account login not working need help now",
    ],
    'Hardware': [
        "laptop screen is busted pls send new one",
        "printer jammed again ugh",
        "keyboard stopped working out of nowhere",
        "monitor wont turn on at all",
        "charger is broken need replacement asap",
        "mouse is glitching so annoying",
        "my pc wont boot up help",
        "headset mic isnt working on calls",
        "laptop battery dies in like 20 mins",
        "screen keeps flickering nonstop",
    ],
    'HR Support': [
        "need to request time off next week",
        "how do i submit my leave request",
        "boss quit need help with his exit paperwork",
        "maternity leave forms where do i find them",
        "need help updating my direct reports",
        "want to check my remaining pto days",
        "hr forms are confusing pls explain",
        "need approval for my vacation days asap",
        "new hire starting monday need onboarding docs",
        "benefits enrollment deadline is when",
    ],
    'Storage': [
        "mailbox full cant get new emails ugh",
        "need more storage on shared drive asap",
        "outlook says storage full wtf do i do",
        "cant save files drive is full",
        "need my quota increased pls",
        "shared folder ran out of space again",
        "cant upload anymore storage maxed out",
        "email bouncing bc mailbox full",
        "need extra gb on my drive asap",
        "storage alert wont go away help",
        # Edge-case fixes (was misclassified as Administrative rights)
        "cant attach files to emails it says mailbox full",
        "outlook wont let me send attachments storage is full",
        "email quota exceeded cant attach anything",
        "inbox is full cant receive or send attachments",
        "getting storage quota exceeded when attaching files in outlook",
    ],
    'Purchase': [
        "need to order new laptops for new hires",
        "can i get a quote for a monitor",
        "need approval to buy new keyboards",
        "want to order extra chairs for the team",
        "need a po number for this purchase",
        "can u get pricing on a new printer",
        "need to buy licenses for 5 more users",
        "want a quote for office headsets",
        "need to purchase software for the dev team",
        "can we order more monitors this month",
        # Edge-case fixes (monitors/hardware nouns in a purchase context)
        "we need 10 new monitors for the team can u order",
        "requesting purchase order for monitors for our department",
        "team needs new monitors please raise a purchase request",
        "need to procure laptops and monitors for new joiners",
        "please approve budget to buy screens and keyboards for office",
    ],
    'Administrative rights': [
        "need admin rights on my laptop pls",
        "cant install anything need elevated perms",
        "need root access to the server asap",
        "requesting admin for the new deployment",
        "need higher perms to change configs",
        "cant modify settings without admin rights",
        "need sudo access on the linux box",
        "requesting elevated rights for setup",
        "need admin to install dev tools",
        "cant change system settings need perms",
    ],
    'Internal Project': [
        "need to add new opportunity to pipeline",
        "pls update the project tracker asap",
        "need someone added to the migration project",
        "want status update on q3 project",
        "need help with the internal tracking sheet",
        "pls add this deal to our pipeline",
        "need the sales forecast doc updated",
        "want to join the new internal project",
        "need report added to project tracker",
        "whos leading the migration project now",
    ],
    'Miscellaneous': [
        "got a sketchy email looks like phishing",
        "office ac is broken so cold in here",
        "thermostat isnt working pls fix",
        "random email looks fake pls check",
        "conference room is freezing again",
        "got spam email not sure if safe",
        "office too hot ac not working",
        "suspicious link in email pls verify",
        "heating is broken in my office",
        "weird email asking for my password",
    ],
}

def get_natural_examples_df() -> pd.DataFrame:
    """
    Casually-typed training examples for every real category, with light
    variation, so the model has actually seen informal phrasing (not just
    formal/lemmatized corpus text) at training time.
    """
    rows = []
    for cat, examples in _NATURAL_EXAMPLES.items():
        for desc in examples:
            variants = [
                desc,
                desc.capitalize(),
                desc + ' please respond quickly',
                'hi ' + desc,
            ]
            for v in variants:
                rows.append({
                    'text': v,
                    'text_clean': clean_text(normalize_input(v)),
                    'label': cat,
                })
    return pd.DataFrame(rows)
