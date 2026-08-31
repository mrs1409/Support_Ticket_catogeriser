import os, shutil

def download_with_kagglehub():
    try:
        import kagglehub
        path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
        print(f"Downloaded to: {path}")
        os.makedirs('data', exist_ok=True)
        for f in os.listdir(path):
            if f.endswith('.csv'):
                shutil.copy(os.path.join(path, f), 'data/tickets.csv')
                print(f"Copied to data/tickets.csv")
                return True
        return False
    except Exception as e:
        print(f"kagglehub failed: {e}")
        return False

def generate_synthetic_fallback():
    """Generate synthetic data if Kaggle download fails."""
    import pandas as pd, random
    print("Generating synthetic training data as fallback...")

    categories = ['Technical Issue', 'Billing', 'Product Inquiry',
                  'Cancellation Request', 'Refund Request']

    templates = {
        'Technical Issue': [
            "The app keeps crashing when I try to open my profile",
            "I cannot login to my account after the latest update",
            "My software is not working and shows an error message",
            "The system is extremely slow and unresponsive today",
            "I am getting a 500 error on the dashboard page",
        ],
        'Billing': [
            "I was charged twice for my monthly subscription",
            "My invoice shows an incorrect amount this month",
            "I need a copy of my billing statement for tax purposes",
            "There is an unauthorised charge on my account",
            "Please update my payment method to a new credit card",
        ],
        'Product Inquiry': [
            "How do I upgrade to the premium plan",
            "What features are included in the enterprise tier",
            "Can you explain how the reporting module works",
            "I would like to know more about your integration options",
            "Does your product support multi-language content",
        ],
        'Cancellation Request': [
            "I want to cancel my subscription immediately",
            "Please close my account and stop all future charges",
            "I am moving to a competitor and need to cancel",
            "How do I cancel my trial before it converts to paid",
            "I would like to downgrade and then cancel my plan",
        ],
        'Refund Request': [
            "I demand a full refund for last months charge",
            "The product does not work as advertised please refund",
            "I cancelled within the trial period but was still charged",
            "Please process a refund for the duplicate payment",
            "I am not satisfied and want my money back",
        ],
    }

    rows = []
    for _ in range(2000):
        cat = random.choice(categories)
        tmpl = random.choice(templates[cat])
        noise = random.choice(['', ' please help', ' this is urgent', ' thank you'])
        rows.append({
            'Ticket Type': cat,
            'Ticket Subject': tmpl[:40],
            'Ticket Description': tmpl + noise,
        })

    df = pd.DataFrame(rows)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/tickets.csv', index=False)
    print(f"Synthetic dataset saved: {len(df)} rows, {len(categories)} categories")
    return True

if __name__ == '__main__':
    success = download_with_kagglehub()
    if not success:
        generate_synthetic_fallback()

    import pandas as pd
    df = pd.read_csv('data/tickets.csv')
    print(f"\nDataset ready: {df.shape[0]} rows")
    print(f"Columns: {df.columns.tolist()}")
    if 'Ticket Type' in df.columns:
        print(df['Ticket Type'].value_counts())
