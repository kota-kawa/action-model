import spacy
from spacy.tokens import Doc
from spacy.training import Example

nlp = spacy.blank("en")

# Add the NER pipe to the pipeline
ner = nlp.add_pipe("ner")

# Add the labels you want to train
ner.add_label("TOPICT")  # Change this to your desired label

# Prepare training data
sentences = [
    "Could you please write an article on the following Star Wars?",
    "Please summarize your content regarding machine learning.",
    "Please tell me more about New York tour in Word.",
    "Could you please write a text on the following university of north alabama?",
    "Could you please provide information on intel?",

    "Could you please write an article on the following UNA?",
    "Could you please write an article on the following internet security?",
    "Could you please write an article on the following Kiminonawa?",
    "Could you please write an article on the following powerful machine?",
    "Could you please write an article on the following web frame work?",
]

# Define the annotations
annotations = [
    {"entities": [(0, 50, "TOPICT")]},
    {"entities": [(0, 39, "TOPICT")]},
    {"entities": [(0, 25, "TOPICT"), (40, 47, "TOPICT")]},
    {"entities": [(0, 46, "TOPICT")]},
    {"entities": [(0, 39, "TOPICT")]},

    {"entities": [(0, 50, "TOPICT")]},
    {"entities": [(0, 50, "TOPICT")]},
    {"entities": [(0, 50, "TOPICT")]},
    {"entities": [(0, 50, "TOPICT")]},
    {"entities": [(0, 50, "TOPICT")]},

]

# Create a list of Example TOPICTs
examples = []
for sentence, annotation in zip(sentences, annotations):
    doc = nlp.make_doc(sentence)
    example = Example.from_dict(doc, annotation)
    examples.append(example)

# Train the model
optimizer = nlp.initialize()
for i in range(100):
    for example in examples:
        nlp.update([example], sgd=optimizer)

# Save the trained model
nlp.to_disk("TOPICT_model")

# Load the trained model
nlp = spacy.load("TOPICT_model")

# Define the text you want to evaluate
texts = [
        "Could you please write an article on the following Star Wars?",
    "Please summarize your content regarding machine learning.",
    "Please tell me more about New York tour in Word.",
    "Could you please write a text on the following university of north alabama?",
    "Could you please provide information on intel?",
]

# Create a doc TOPICT for each text and print the non-labeled entities

for text in texts:
    doc = nlp(text)
    print(f"Text: {text}")
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")
    print("\n")