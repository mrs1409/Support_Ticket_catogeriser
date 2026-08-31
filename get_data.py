import os, shutil, urllib.request, zipfile

def try_kagglehub():
    # NOTE: suraj520/customer-support-ticket-dataset has Ticket Description text
    # that is templated boilerplate with NO correlation to Ticket Type (confirmed:
    # every model trains to ~20% accuracy = random chance on 5 classes). Using
    # adisongoh/it-service-ticket-classification-dataset instead: real IT service
    # desk tickets where text genuinely predicts the topic category (~86% accuracy
    # sanity-checked with a plain LogisticRegression baseline).
    try:
        import kagglehub
        print("Trying kagglehub...")
        path = kagglehub.dataset_download("adisongoh/it-service-ticket-classification-dataset")
        print(f"Downloaded to: {path}")
        os.makedirs('data', exist_ok=True)
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.csv'):
                    src = os.path.join(root, f)
                    shutil.copy(src, 'data/tickets.csv')
                    print(f"Copied: {f} -> data/tickets.csv")
                    return True
        return False
    except Exception as e:
        print(f"kagglehub failed: {e}")
        return False

def try_kaggle_cli():
    try:
        import subprocess
        print("Trying kaggle CLI...")
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d',
             'adisongoh/it-service-ticket-classification-dataset',
             '--unzip', '-p', 'data/'],
            capture_output=True, text=True, timeout=120
        )
        print(result.stdout)
        if result.returncode == 0:
            for f in os.listdir('data'):
                if f.endswith('.csv'):
                    src = os.path.join('data', f)
                    if src != 'data/tickets.csv':
                        shutil.copy(src, 'data/tickets.csv')
                    print(f"Dataset ready: data/tickets.csv")
                    return True
        print(f"CLI error: {result.stderr}")
        return False
    except Exception as e:
        print(f"CLI failed: {e}")
        return False

def make_rich_synthetic():
    import pandas as pd
    import random
    print("Generating rich synthetic dataset (5000 rows, 5 categories)...")
    random.seed(42)

    categories = {
        'Technical Issue': [
            "My application crashes every time I try to open the dashboard",
            "I am getting error code 500 when I try to submit the form",
            "The software is not responding and I cannot do anything",
            "Login page shows a blank white screen after entering credentials",
            "API integration is returning null values for all requests",
            "Cannot install the latest version keeps showing dependency error",
            "Mobile app freezes completely after the recent update",
            "Database connection times out after exactly 30 seconds",
            "SSL certificate error appearing on all pages of the platform",
            "Two factor authentication code is not being delivered to my phone",
            "File upload feature broken shows progress bar then nothing happens",
            "Search functionality returns wrong results since last Tuesday",
            "Export to PDF option is completely missing from the menu now",
            "System performance extremely slow taking 5 minutes to load a page",
            "Integration with Zapier stopped working after the platform update",
            "Cannot reset my password the reset email link expires immediately",
            "Video calls keep dropping after exactly 10 minutes of connection",
            "Data sync between mobile and desktop is not working properly",
            "Charts and graphs not loading just showing a spinning icon",
            "Webhook notifications stopped firing completely three days ago",
        ],
        'Billing Inquiry': [
            "I was charged twice for my monthly subscription this billing cycle",
            "My invoice shows an amount that does not match what I agreed to pay",
            "I need a proper VAT invoice for my company accounting records",
            "Can you explain what the additional line item on my bill is for",
            "I cancelled before the renewal date but was still charged the full amount",
            "My credit card was declined but the money was taken from my account",
            "I need to update my billing address to match my new company location",
            "How do I switch from monthly to annual billing to get the discount",
            "I was promised a promotional rate but am being charged the full price",
            "My company requires a purchase order number on all invoices",
            "The discount code I applied is not showing up on my final invoice",
            "I need to split my subscription payment across two different cards",
            "There is an unauthorized charge on my account I did not approve",
            "Can I get a prorated refund for the remaining days of my subscription",
            "My subscription auto renewed even though I turned off auto renewal",
            "I need a detailed breakdown of all charges for the last six months",
            "The annual plan was debited but my account still shows monthly plan",
            "I changed my payment method but the old card was still charged",
            "How do I get a receipt for my payment for expense report purposes",
            "I am being charged for users that I already removed from my account",
        ],
        'Product Inquiry': [
            "Does your platform support integration with Salesforce CRM",
            "What is the maximum file size I can upload to the system",
            "Can you explain how the automated workflow feature works in detail",
            "Is there a way to export all my data in bulk to a CSV format",
            "Does the enterprise plan include dedicated customer support",
            "How many team members can I add to a single workspace",
            "Is the mobile app available for both iOS and Android devices",
            "What programming languages does your API support for integration",
            "Can I white label the platform with my own company branding",
            "Does your service comply with GDPR and data protection regulations",
            "Is there an offline mode available when there is no internet connection",
            "What are the storage limits for each tier of the subscription plan",
            "Can multiple users work on the same document at the same time",
            "Does the reporting module support custom date range filtering",
            "Is there a free trial available before committing to a paid plan",
            "How does the AI recommendation engine generate its suggestions",
            "Can I set different permission levels for different team members",
            "Does your platform have an open API for custom integrations",
            "What kind of data backup and recovery options do you provide",
            "Is single sign on SSO available for enterprise plan customers",
        ],
        'Cancellation Request': [
            "I would like to cancel my subscription effective immediately please",
            "Please close my account and ensure no future charges are made",
            "I am moving to a competitor and need to cancel my plan today",
            "I want to downgrade from enterprise to the free tier immediately",
            "Please cancel my annual subscription and process a partial refund",
            "I no longer need this service and want to close everything down",
            "How do I cancel before my trial converts to a paid subscription",
            "I am cancelling because the pricing has become too expensive for us",
            "Please cancel and make sure my data is completely deleted as well",
            "I want to pause my subscription for three months not cancel fully",
            "The features we needed are not available so we are cancelling",
            "Please cancel the account for the user who has left our company",
            "I signed up by mistake and want to cancel within the cooling period",
            "We are shutting down our business and need all accounts cancelled",
            "Can I cancel just the add on features and keep the base plan",
            "I would like to cancel and transfer my data to my personal account",
            "Please cancel my subscription as we failed our funding round",
            "I am retiring and will no longer need this service going forward",
            "Can I cancel mid cycle and get a refund for the unused portion",
            "Please cancel immediately and send me a confirmation email",
        ],
        'Refund Request': [
            "I demand a full refund for the payment made last week immediately",
            "The product does not work as advertised and I want my money back",
            "I was charged after cancelling and need this refunded right away",
            "Please refund the duplicate charge that appeared on my statement",
            "I am not satisfied with the service quality and want a full refund",
            "The feature I paid for the premium plan for does not even work",
            "I accidentally upgraded to the wrong plan please refund the difference",
            "I have been waiting three weeks for a refund that was promised to me",
            "The annual plan was charged but I only wanted the monthly option",
            "Please issue a refund as we never used the service after signing up",
            "I cancelled within the 30 day money back guarantee period for refund",
            "The billing team promised me a refund two weeks ago where is it",
            "I need a refund processed before end of month for my budget cycle",
            "Wrong pricing was shown on your website please refund the difference",
            "The service was down for two days last month I want a credit refund",
            "Please refund the charge for the additional seats I never requested",
            "I upgraded and downgraded same day please refund the upgrade cost",
            "There was a technical error during payment that charged me twice",
            "I was given a free trial extension promise but was still charged",
            "Please process urgent refund my company finance team is asking",
        ],
    }

    noise_prefixes = [
        "", "Hi there, ", "Hello, ", "Good morning, ", "Dear support team, ",
        "I need help. ", "Urgent: ", "Please help. ", "To whom it may concern, ", "",
    ]
    noise_suffixes = [
        "", " Please help.", " This is urgent.", " Thank you.",
        " Awaiting your response.", " Please resolve ASAP.", "",
        " This is unacceptable.", " I need this fixed today.", "",
    ]
    subjects = {
        'Technical Issue': ["App not working", "System error", "Bug report",
                           "Technical problem", "Cannot access", "Error on platform",
                           "Software issue", "Login problem", "Feature broken"],
        'Billing Inquiry': ["Invoice question", "Billing issue", "Charge query",
                           "Payment problem", "Invoice needed", "Billing help",
                           "Subscription charge", "Payment question"],
        'Product Inquiry': ["Feature question", "Product info", "How does this work",
                           "Integration question", "Plan details", "Capability query"],
        'Cancellation Request': ["Cancel subscription", "Close account",
                                "Cancel my plan", "Downgrade request", "Cancel immediately"],
        'Refund Request': ["Refund needed", "Request refund", "Money back",
                          "Refund my payment", "Charge refund", "Urgent refund"],
    }

    rows = []
    for _ in range(5000):
        cat = random.choice(list(categories.keys()))
        base = random.choice(categories[cat])
        prefix = random.choice(noise_prefixes)
        suffix = random.choice(noise_suffixes)
        description = prefix + base + suffix
        subject = random.choice(subjects[cat])
        rows.append({
            'Ticket Type': cat,
            'Ticket Subject': subject,
            'Ticket Description': description,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/tickets.csv', index=False)
    print(f"Synthetic dataset: {len(df)} rows, {df['Ticket Type'].nunique()} categories")
    print(df['Ticket Type'].value_counts())
    return True

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    success = try_kagglehub()
    if not success:
        success = try_kaggle_cli()
    if not success:
        make_rich_synthetic()

    import pandas as pd
    df = pd.read_csv('data/tickets.csv')
    print(f"\nFINAL: {df.shape[0]} rows, columns: {df.columns.tolist()}")
