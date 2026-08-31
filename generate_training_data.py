"""
generate_training_data.py
Generates high-quality synthetic training data with realistic, category-specific
ticket text and rich filler variation. Produces 8,000 unique rows across 5 categories.

Run: python generate_training_data.py
Output: data/tickets.csv
"""
import pandas as pd
import random
import os

random.seed(42)

# ─── FILLER POOLS (large for maximum combination variety) ─────────────────────
AMOUNTS       = ['$9.99', '$14.99', '$19.99', '$24.99', '$29.00', '$39.99',
                 '$49.99', '$59.99', '$79.99', '$99.99', '$149.00', '$199.00']
DATES         = ['January 3', 'January 15', 'February 7', 'February 22',
                 'March 4', 'March 18', 'April 1', 'April 29', 'May 6',
                 'May 21', 'June 10', 'June 28', 'last Monday', 'last Friday',
                 'two weeks ago', 'yesterday', 'this morning']
MONTHS        = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
PLAN_PRICES   = ['$9.99', '$14.99', '$19.99', '$29.00', '$39.99']
CODES         = ['E404', 'E500', 'E1023', 'AUTH_FAIL', '403', 'ERR_CONN',
                 '502', 'E2001', 'TIMEOUT', 'SESSION_EXPIRED']
PCTS          = ['23', '45', '67', '78', '80', '91', '95', '99']
DAYS          = ['2', '3', '5', '7', '10', '14', '21', '30']
FEATURES      = ['dashboard', 'settings page', 'reports module', 'export tool',
                 'profile editor', 'billing section', 'admin panel',
                 'analytics view', 'team management', 'notifications center',
                 'calendar view', 'document editor', 'search function',
                 'API playground', 'integration manager']
TOOLS         = ['Slack', 'Zapier', 'Salesforce', 'Google Sheets', 'HubSpot',
                 'Trello', 'Jira', 'Microsoft Teams', 'Zendesk', 'Shopify',
                 'QuickBooks', 'Mailchimp', 'Intercom', 'GitHub', 'Asana']
DEVICES       = ['iPhone 15', 'iPhone 14 Pro', 'Samsung Galaxy S24',
                 'Google Pixel 8', 'iPad Pro', 'Android tablet',
                 'MacBook Pro', 'Windows laptop', 'Surface Pro']
OS_VERSIONS   = ['iOS 17', 'iOS 16', 'Android 14', 'Android 13',
                 'Windows 11', 'macOS Sonoma', 'macOS Ventura', 'Ubuntu 22']
PLANS         = ['Starter', 'Basic', 'Professional', 'Business', 'Enterprise',
                 'Team', 'Growth', 'Scale', 'Advanced']
COUPON_CODES  = ['SAVE10', 'WELCOME20', 'PROMO15', 'ANNUAL30', 'FIRST50']
NAMES         = ['John', 'Sarah', 'Michael', 'Emma', 'David', 'Lisa']

FILLERS = {
    'amount':       AMOUNTS,
    'date':         DATES,
    'month':        MONTHS,
    'plan_price':   PLAN_PRICES,
    'pct':          PCTS,
    'code':         CODES,
    'days':         DAYS,
    'feature':      FEATURES,
    'tool':         TOOLS,
    'device':       DEVICES,
    'os':           OS_VERSIONS,
    'plan':         PLANS,
    'coupon':       COUPON_CODES,
    'name':         NAMES,
}

# ─── TEMPLATES ────────────────────────────────────────────────────────────────
TEMPLATES = {
    'Billing inquiry': [
        "I was charged {amount} on {date} but I did not authorise this payment.",
        "My invoice for {month} shows {amount} but my plan should be {plan_price} per month.",
        "I received two separate charges of {amount} on the same day. Please explain.",
        "I cannot find my billing statement for {month}. Please send me a copy by email.",
        "I need to update my payment method. My old card expired and I have a new one.",
        "My credit card was declined but I still seem to be subscribed. Was I charged?",
        "Can you send me a VAT receipt for the {amount} charge that appeared on {date}?",
        "I was billed {amount} instead of {plan_price} which is my current plan rate.",
        "I upgraded my plan but was still charged the old rate of {amount} this month.",
        "My bank shows a debit from your company for {amount}. What exactly is this for?",
        "Please change my billing cycle from monthly to annual billing.",
        "I have not received any invoice this month. Could you resend it to my email?",
        "I need a complete breakdown of all charges from the last three months.",
        "I see a {amount} charge labelled as an add-on fee. I never requested this.",
        "My coupon code {coupon} was not applied. I should have received a discount.",
        "I was charged {amount} immediately after signing up for a free trial.",
        "My plan renews on {date} but I was charged early on {date}.",
        "Can I switch from credit card to bank transfer for my {amount} monthly payments?",
        "I received a chargeback notification for {amount}. Please clarify this charge.",
        "My account shows a pending charge of {amount} that I don't recognise.",
        "I was on the {plan} plan at {plan_price} but the invoice says {amount}.",
        "Please provide an official tax invoice for all payments made in {month}.",
        "I was charged {amount} for a feature I never activated or used.",
        "Why was I charged {amount} when I cancelled before the renewal date?",
        "I need to dispute the {amount} charge from {date} as it was not authorised.",
    ],
    'Technical issue': [
        "The app crashes every time I try to open the {feature} section on my {device}.",
        "I keep getting error code {code} when attempting to log in to my account.",
        "The {feature} is completely blank and not loading anything at all.",
        "After the latest update, the {feature} stopped working entirely.",
        "File uploads fail every time. The progress bar gets stuck at {pct}%.",
        "The website is extremely slow today. Every page takes over 30 seconds to load.",
        "I am completely locked out of my account. The password reset email never arrives.",
        "Two-factor authentication is not sending me the verification code via SMS.",
        "My data is not syncing at all between the {device} app and the web version.",
        "I keep getting a 500 server error whenever I try to access the main dashboard.",
        "The PDF export feature generates a corrupted file that cannot be opened.",
        "I cannot connect the integration with {tool}. It always returns an authentication error.",
        "The search function returns zero results even when I type an exact match.",
        "Push notifications stopped working on {device} after I updated to {os}.",
        "The {feature} shows all events on the wrong dates. The calendar is completely off.",
        "I cannot log in with my Google account. It just redirects me in a loop.",
        "The API returns a {code} error on every single request I make.",
        "My account was working fine and then suddenly locked me out with error {code}.",
        "The {feature} on {device} running {os} crashes with no error message.",
        "Data I saved last week is completely gone. The {feature} shows empty.",
        "The browser version works but the mobile app on {device} gives error {code}.",
        "I cannot change my password. The form submits but nothing changes.",
        "The {tool} integration was working last week but stopped syncing today.",
        "Video playback inside the {feature} buffers constantly and never loads.",
        "I am seeing someone else's data in my {feature}. This is a serious bug.",
    ],
    'Product inquiry': [
        "What is the difference between the {plan} plan and the basic plan?",
        "Does your product support a native integration with {tool}?",
        "How many users can I add to a single account on the {plan} plan?",
        "Is there a free trial available before committing to a paid subscription?",
        "How does the {feature} work exactly? I need to understand before upgrading.",
        "Can I export all my data if I decide to switch to a different provider?",
        "Do you offer annual billing and does it come with a discount compared to monthly?",
        "What security certifications and compliance standards does your platform hold?",
        "I am considering the {plan} plan. Can you walk me through the included features?",
        "Does the enterprise tier include a dedicated account manager?",
        "Can your API be used to pull automated reports on a scheduled basis?",
        "What is the data retention and backup policy on the {plan} plan?",
        "Is there an official mobile app available for both iOS and Android platforms?",
        "How do I configure single sign-on for my entire team of 50 users?",
        "Can I use your platform under my own brand for client-facing use?",
        "What happens to my data if I downgrade from the {plan} to a lower tier?",
        "Is there a limit on the number of API calls I can make per month?",
        "How does the {feature} compare to what competitors offer in this space?",
        "Can two team members work on the same document simultaneously?",
        "What is your uptime guarantee and what SLA do you offer for {plan} plans?",
        "Do you have a referral program and how does the commission structure work?",
        "Is the {plan} plan suitable for a team of {pct} people working remotely?",
        "How long does it take to onboard a new team onto your {plan} plan?",
        "What file formats does the {feature} support for import and export?",
        "Can I customise user permissions and access levels for different team members?",
    ],
    'Cancellation request': [
        "Please cancel my subscription immediately. I no longer require this service.",
        "I want to close my account permanently and have all my personal data deleted.",
        "How do I cancel before the next billing date to avoid being charged again?",
        "I am switching to a competitor product. Please cancel my account today.",
        "Please cancel my free trial before it converts to a paid subscription.",
        "I have been trying to cancel for {days} days but the cancel button does not work.",
        "I want to downgrade to the free tier now and cancel the paid plan entirely.",
        "Please send me a written cancellation confirmation to my registered email.",
        "I cancelled {days} days ago but was still charged {amount} this month.",
        "My entire team has stopped using your product. Please cancel all {pct} accounts.",
        "I am closing my business and need to cancel all active subscriptions immediately.",
        "Please cancel and confirm there will be no further charges to my card.",
        "I want to cancel but need to download and keep a copy of all my data first.",
        "Can I pause my subscription for {days} weeks instead of cancelling completely?",
        "I want to cancel only the add-on and keep the base subscription active.",
        "My company was acquired and we need to cancel all existing contracts.",
        "Please cancel as of {date} and process a pro-rata refund for unused time.",
        "I signed up by mistake and want to cancel within the {days}-day window.",
        "The service has been down for {days} days and I want to cancel due to this.",
        "I no longer need the {plan} plan features. Please cancel and downgrade me.",
        "Please cancel auto-renewal on my account but keep it active until {date}.",
        "I am moving countries and your service is not available in my new location.",
        "My budget has been cut and I need to cancel all premium services immediately.",
        "I have found a better solution for my needs. Please process my cancellation.",
        "Cancel my subscription and delete my data in compliance with GDPR please.",
    ],
    'Refund Request': [
        "I want a full refund for the {amount} charge that was made on {date}.",
        "I was charged {amount} after my cancellation date. Please refund this.",
        "I am not satisfied with the product quality and want my money back.",
        "Please refund the {amount} that was charged to my account in error.",
        "I accidentally subscribed to the wrong plan. Please refund the {amount} difference.",
        "My refund of {amount} has not been credited back after {days} business days.",
        "I upgraded my plan by mistake. I want the difference of {amount} refunded.",
        "I was charged {amount} during what was supposed to be a free trial period.",
        "The product does not match what was advertised at all. I want a full refund.",
        "I am within the 30-day money-back guarantee window. Please refund {amount}.",
        "I was billed twice in the same month. Please reverse one charge of {amount}.",
        "I cancelled before the renewal date but was charged {amount} anyway. Refund please.",
        "I never received login access to the product I paid {amount} for.",
        "Please process a reversal for the disputed {amount} charge immediately.",
        "I have a payment receipt for {amount} but I never used the service at all.",
        "The feature I paid {amount} extra for was removed without any notice.",
        "I requested a cancellation on {date} but was charged again on {date}.",
        "I received a corrupted file that I paid {amount} to download. Please refund.",
        "Your service caused data loss and I am requesting a full refund of {amount}.",
        "I was promised a refund of {amount} by your support agent {days} days ago.",
        "I downgraded my plan but was still charged the higher rate of {amount}.",
        "The {amount} annual payment was taken without prior notice or email warning.",
        "I disputed a charge of {amount} with my bank. Please process the refund directly.",
        "My company policy changed and I need a refund of {amount} for unused months.",
        "I was charged {amount} for a renewal I explicitly cancelled {days} days ago.",
    ],
}

SUBJECTS = {
    'Billing inquiry':       ['Billing question', 'Invoice issue', 'Charge problem',
                               'Payment error', 'Billing statement needed',
                               'Incorrect charge', 'Payment method update'],
    'Technical issue':       ['App not working', 'Login error', 'Bug report',
                               'System error', 'Feature broken', 'Crash report',
                               'Integration error', 'Sync problem'],
    'Product inquiry':       ['Plan comparison', 'Feature question', 'Upgrade query',
                               'Integration query', 'General information',
                               'Pricing question', 'Onboarding help'],
    'Cancellation request':  ['Cancel subscription', 'Close account', 'Stop service',
                               'Unsubscribe request', 'Cancel plan', 'Account closure'],
    'Refund Request':        ['Refund request', 'Money back', 'Charge dispute',
                               'Duplicate charge refund', 'Billing error refund'],
}

NOISE = [
    '',
    '',
    '',
    ' Please help.',
    ' This is urgent.',
    ' Thank you for your assistance.',
    ' I have been a loyal customer.',
    ' I expect a prompt response.',
    ' I look forward to hearing from you.',
    ' Please resolve this as soon as possible.',
]


def fill(template: str) -> str:
    """Replace all {placeholder} tokens with random values from filler pools."""
    for key, options in FILLERS.items():
        placeholder = '{' + key + '}'
        while placeholder in template:
            template = template.replace(placeholder, random.choice(options), 1)
    return template


def generate(rows_per_category: int = 1600) -> pd.DataFrame:
    rows = []
    for cat, templates in TEMPLATES.items():
        subjects = SUBJECTS[cat]
        count = 0
        safety = 0
        while count < rows_per_category and safety < rows_per_category * 50:
            safety += 1
            tmpl = random.choice(templates)
            desc = fill(tmpl) + random.choice(NOISE)
            subj = random.choice(subjects)
            rows.append({
                'Ticket Subject':     subj,
                'Ticket Description': desc,
                'Ticket Type':        cat,
                'Ticket Priority':    random.choice(['Low', 'Medium', 'High', 'Critical']),
                'Ticket Status':      random.choice(['Open', 'In Progress', 'Closed']),
            })
            count += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df['Ticket ID'] = range(1, len(df) + 1)
    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    output = 'data/tickets.csv'

    df = generate(rows_per_category=1600)
    df.to_csv(output, index=False)

    print(f"Generated: {output}")
    print(f"Shape: {df.shape}")
    print(df['Ticket Type'].value_counts())

    # Verify uniqueness
    dup_count = df['Ticket Description'].duplicated().sum()
    print(f"Duplicate descriptions: {dup_count}")
    print(f"\nSample row:")
    row = df.iloc[0]
    print(f"  Subject: {row['Ticket Subject']}")
    print(f"  Desc:    {row['Ticket Description']}")
    print(f"  Label:   {row['Ticket Type']}")
