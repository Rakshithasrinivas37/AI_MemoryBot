class LongTerm:
    def __init__(self):
        self.summary = ""

    def update_memory(self, summary):
        self.summary = summary

    def get(self):
        return self.summary

    def clear(self):
        self.summary = ""

    def is_empty(self):
        return self.summary == ""