class Track:
    def __init__(self, id, bbox, features, max_age, min_hits):
        self.id = id
        self.bbox = bbox
        self.features = features
        self.age = 0
        self.hits = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.active = False

    def update(self, bbox, features):
        self.bbox = bbox
        self.features = features
        self.age = 0
        if self.hits < self.min_hits:
            self.hits += 1
        if self.hits == self.min_hits:
            self.active = True

    def miss_detection(self):
        self.age += 1
        self.hits = 0
        if self.age > self.max_age:
            self.active = False

    def activate(self):
        self.hits = self.min_hits
        self.active = True
