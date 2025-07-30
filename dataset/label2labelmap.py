from enum import Enum

class GENDER(Enum):
    MALE = (0, "Male")
    FEMALE = (1, "Female")
    
    def __init__(self, id, label):
        self.id = id
        self.label = label
    
    @classmethod
    def get_label(cls, id):
        for gender in cls:
            if gender.id == id:
                return gender.label
        return None

class EMOTION(Enum):
    ANGRY = (0, "Angry")
    DISGUST = (1, "Disgust")
    FEAR = (2, "Fear")
    HAPPY = (3, "Happy")
    NEUTRAL = (4, "Neutral")
    SAD = (5, "Sad")
    SURPRISE = (6, "Surprise")
    
    def __init__(self, id, label):
        self.id = id
        self.label = label
    
    @classmethod
    def get_label(cls, id):
        for emotion in cls:
            if emotion.id == id:
                return emotion.label
        return None

class AGE_GROUP(Enum):
    AGE_0_2 = (0, "0-2")
    AGE_3_9 = (1, "3-9")
    AGE_10_19 = (2, "10-19")
    AGE_20_29 = (3, "20-29")
    AGE_30_39 = (4, "30-39")
    AGE_40_49 = (5, "40-49")
    AGE_50_59 = (6, "50-59")
    AGE_60_69 = (7, "60-69")
    AGE_MORE_THAN_70 = (75, "more than 70")
    
    def __init__(self, id, label):
        self.id = id
        self.label = label
    
    @classmethod
    def get_label(cls, id):
        for age_group in cls:
            if age_group.id == id:
                return age_group.label
        return None