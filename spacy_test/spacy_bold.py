import spacy
from spacy.tokens import Doc
from spacy.training import Example

nlp = spacy.blank("en")

# Add the NER pipe to the pipeline
ner = nlp.add_pipe("ner")

# Add the labels you want to train
ner.add_label("BOLD")  # Change this to your desired label

# Prepare training data
sentences = [
    "Make the part Tomorrow's plans in bold.",
    "Highlight the section below and select 'Internet Connection'.",
    "Please make the 'pineapple' part bold.",
    "Please make 'San Francisco' bold.",
    "Emphasis on 'college restaurant'.",
    "Please display the following text in bold. Flower garden.",

    "Make the part Star Wars in bold.",
    "Make the part franchise created by in bold.",
    "Make the part 1977 film and in bold.",
    "Make the part franchises of all time in bold.",
    "Make the part quickly in bold.",
]

# Define the annotations
annotations = [
    {"entities": [(0, 13, "BOLD"), (31, 39, "BOLD")]},
    {"entities": [(0, 38, "BOLD")]},
    {"entities": [(0, 15, "BOLD"), (28, 38, "BOLD")]},
    {"entities": [(0, 11, "BOLD"), (28, 32, "BOLD")]},
    {"entities": [(0, 11, "BOLD")]},
    {"entities": [(0, 42, "BOLD")]},

    {"entities": [(0, 13, "BOLD"), (24, 31, "BOLD")]},
    {"entities": [(0, 13, "BOLD"), (32, 42, "BOLD")]},
    {"entities": [(0, 13, "BOLD"), (28, 35, "BOLD")]},
    {"entities": [(0, 13, "BOLD"), (37, 44, "BOLD")]},
    {"entities": [(0, 13, "BOLD"), (22, 29, "BOLD")]},
]

# Create a list of Example objects
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
nlp.to_disk("BOLD_model")


# Load the trained model
nlp = spacy.load("BOLD_model")

# Define the text you want to evaluate
texts = [
    "Make the part Tomorrow's plans in bold.",
    "Highlight the section below and select 'Internet Connection'.",
    "Please make the 'pineapple' part bold.",
    "Please make 'San Francisco' bold.",
    "Emphasis on 'college restaurant'.",
    "Please display the following text in bold. Flower garden.",
]

# Create a doc object for each text and print the entities
for text in texts:
    doc = nlp(text)
    print(f"Text: {text}")
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")
    print("\n")

