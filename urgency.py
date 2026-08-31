from typing import Tuple

HIGH_KEYWORDS = {
    'urgent': 10, 'asap': 10, 'immediately': 10, 'emergency': 10,
    'critical': 10, 'outage': 10, 'not working': 9, 'cannot access': 9,
    'cant access': 9, "can't access": 9, 'locked out': 9, 'locked': 7,
    'data loss': 10, 'breach': 10, 'hacked': 10, 'charged twice': 9,
    'double charge': 9, 'down': 8, 'crash': 9, 'crashed': 9,
    'broken': 8, 'failure': 8, 'security': 8, 'wrong charge': 8,
    'deadline': 8, 'escalate': 9, 'legal': 8, 'lawsuit': 10,
    'threatening': 8, 'overdue': 7, 'not responding': 8,
    'completely broken': 10, 'totally broken': 10, 'system down': 10,
    'data lost': 10, 'corrupted': 9, 'deleted': 7, 'unauthorized': 9,
    'fraud': 10, 'stolen': 9, 'compromised': 9, 'exposed': 8,
}

MEDIUM_KEYWORDS = {
    'error': 5, 'issue': 4, 'problem': 4, 'slow': 4, 'delay': 5,
    'not received': 5, 'missing': 5, 'incorrect': 5, 'wrong': 4,
    'complaint': 5, 'disappointed': 4, 'frustrating': 5,
    'still waiting': 5, 'follow up': 4, 'billing': 4, 'invoice': 4,
    'payment': 4, 'charge': 4, 'need help': 3, 'please help': 3,
    'update': 3, 'status': 3, 'refund': 6, 'cancel': 4,
    'not working': 6, 'does not work': 6, 'stopped working': 7,
    'bug': 5, 'glitch': 5, 'malfunction': 6, 'failed': 5,
    'timeout': 5, 'disconnected': 5, 'not loading': 6, 'blank': 4,
}

LOW_KEYWORDS = {
    'question': 2, 'how do i': 2, 'how to': 2, 'wondering': 1,
    'curious': 1, 'information': 1, 'feedback': 2, 'suggestion': 2,
    'feature request': 2, 'upgrade': 2, 'account': 1, 'learn': 1,
    'understand': 1, 'know more': 1, 'interested in': 1,
    'would like to': 1, 'can you explain': 2, 'does your': 1,
}

def classify_urgency(text: str) -> Tuple[str, str]:
    text_lower = text.lower()

    high_matched = [(kw, w) for kw, w in HIGH_KEYWORDS.items() if kw in text_lower]
    medium_matched = [(kw, w) for kw, w in MEDIUM_KEYWORDS.items() if kw in text_lower]
    low_matched = [(kw, w) for kw, w in LOW_KEYWORDS.items() if kw in text_lower]

    # Level is set by the highest-severity tier that actually matched, not a
    # cumulative sum — otherwise several low/medium keywords stacking together
    # could outscore a genuinely urgent ticket with just one high keyword.
    if high_matched:
        level, matched = 'High', high_matched
    elif medium_matched:
        level, matched = 'Medium', medium_matched
    elif low_matched:
        level, matched = 'Low', low_matched
    else:
        level, matched = 'Low', []

    top = sorted(matched, key=lambda x: -x[1])[:3]
    reason = (
        'Triggered by: ' + ', '.join(f'"{kw}" (+{w})' for kw, w in top)
        if top else 'No urgency keywords found'
    )
    return level, reason

def urgency_color(urgency: str) -> str:
    return {
        'High': '#dc2626',
        'Medium': '#d97706',
        'Low': '#16a34a'
    }.get(urgency, '#6b7280')
