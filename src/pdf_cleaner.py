import re

class PDFCleaner:
    def __init__(self, text: str):
        self.text = text

    def remove_copyrights(self):
        """Removes © blocks, license info, and repeated footer notes."""
        self.text = re.sub(r'©.*?(?=\n)', '', self.text, flags=re.DOTALL)
        self.text = re.sub(r'Readers may use this work.*?unaltered\.', '', self.text, flags=re.DOTALL)
        self.text = re.sub(r'Requests to reuse, adapt, or distribute.*?@diabetes\.org\.', '', self.text, flags=re.DOTALL)

    def remove_urls(self):
        """Removes URLs from text to avoid noise."""
        self.text = re.sub(r'https?://\S+', '', self.text)

    def fix_hyphenation(self):
        """Fix words split across lines with hyphenation (e.g., hypergly-\ncemia → hyperglycemia)."""
        self.text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', self.text)

    def remove_headers_footers(self, header_patterns: list = None):
        """Removes repeated headers or footers across pages."""
        if header_patterns:
            for pattern in header_patterns:
                self.text = re.sub(pattern, '', self.text)
        # Remove common repeated short headers/footers automatically
        self.text = re.sub(r'^\s*(Diabetes Care|American Diabetes Association|@AmericanDiabetesAssociation).*$', '', self.text, flags=re.MULTILINE)

    def normalize_whitespace(self):
        """Strips extra newlines, multiple spaces, tabs, etc."""
        self.text = re.sub(r'\n+', '\n', self.text)
        self.text = re.sub(r'[ \t]+', ' ', self.text)
        self.text = self.text.strip()

    def remove_references(self):
        """Removes citations, page numbers, section numbers, and S-codes."""
        # square bracket references [1], [23]
        self.text = re.sub(r'\[\d+\]', '', self.text)
        # parenthesis references (Smith et al., 2020), (2025)
        self.text = re.sub(r'\([A-Za-z]+ et al\., \d{4}\)', '', self.text)
        self.text = re.sub(r'\(\d{4}\)', '', self.text)
        # page numbers / S numbers
        self.text = re.sub(r'\bS\d+\b', '', self.text)
        self.text = re.sub(r'\bp\. \d+\b', '', self.text)
        # section numbers like "2. DIAGNOSIS AND CLASSIFICATION"
        self.text = re.sub(r'^\d+\.\s+.*$', '', self.text, flags=re.MULTILINE)

    def general_regex_clean(self, patterns: list):
        """Removes any other noise patterns given by regex."""
        for pattern in patterns:
            self.text = re.sub(pattern, '', self.text)

    def clean_all(self, header_patterns: list = None, extra_patterns: list = None):
        """Runs all cleaning methods in order."""
        self.remove_copyrights()
        self.remove_urls()
        self.fix_hyphenation()
        self.remove_headers_footers(header_patterns)
        self.remove_references()
        if extra_patterns:
            self.general_regex_clean(extra_patterns)
        self.normalize_whitespace()
        return self.text