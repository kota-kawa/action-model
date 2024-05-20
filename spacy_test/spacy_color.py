import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random 

# Load the spacy model
nlp = spacy.load('en_core_web_sm')

# Get the Named Entity Recognizer
ner = nlp.get_pipe("ner")

# Add a new entity label to entity recognizer
if "COLOR" not in ner.labels:
    ner.add_label("COLOR")

# Training examples in the required format
# Training examples in the required format
TRAIN_DATA = [
    ("The sky is blue.", {"entities": [(11, 15, "COLOR")]}),
    ("I have a red car.", {"entities": [(9, 12, "COLOR")]}),
    ("She wore a yellow dress.", {"entities": [(11, 17, "COLOR")]}),
    ("The leaves are green.", {"entities": [(15, 20, "COLOR")]}),
    ("I bought a pink shirt.", {"entities": [(11, 15, "COLOR")]}),
    ("He has brown eyes.", {"entities": [(7, 12, "COLOR")]}),
    ("The sun is orange.", {"entities": [(11, 17, "COLOR")]}),
    ("She likes purple flowers.", {"entities": [(10, 16, "COLOR")]}),
    ("I want a white hat.", {"entities": [(9, 14, "COLOR")]}),
    ("He found a black cat.", {"entities": [(11, 16, "COLOR")]}),
    ("Your green house is so nice.", {"entities": [(5, 10, "COLOR")]}),
    # Add more examples here
]


# Get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.resume_training()
    for itn in range(100):   # Number of iterations can be changed
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], sgd=optimizer, drop=0.35, losses=losses)

# Save the model
nlp.to_disk("model")

# Load the trained model
nlp = spacy.load("model")

doc = nlp("""The sky is blue and the car is red. And I love yellow flowers.Do you like green? 
          Also pink peach is very delicious. I think brown hair is so nice.""")
colors = [ent.text for ent in doc.ents if ent.label_ == "COLOR"]
#1つ
#colors = next((ent.text for ent in doc.ents if ent.label_ == "COLOR"), None)


print(colors)  # Output: ['blue', 'red']
