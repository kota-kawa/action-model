import spacy
from spacy.tokens import Doc
from spacy.training import Example

nlp = spacy.blank("en")

# Add the NER pipe to the pipeline
ner = nlp.add_pipe("ner")

# Add the labels you want to train
ner.add_label("FORM")  # Change this to your desired label

# Prepare training data
sentences = [
    "Please enter details into the form.",
    "Enter SF movies into the form.",
    "Could you fill in the form with programming data?",
    "Kindly provide good restaurant in the form.",
    "Make sure to put soccer practice in the form.",

    "Please enter Star Wars into the form.",
    "Please enter phenomenon into the form.",
    "Please enter George Lucas into the form.",
    "Please enter film and quickly into the form.",
    "Please enter created by into the form.",
]

# Define the annotations
annotations = [
    {"entities": [(0, 12, "FORM"), (21, 34, "FORM")]},
    {"entities": [(0, 5, "FORM"), (16, 29, "FORM")]},
    {"entities": [(0, 31, "FORM")]},
    {"entities": [(0, 14, "FORM"), (31, 42, "FORM")]},
    {"entities": [(0, 16, "FORM"), (31, 44, "FORM")]},

    {"entities": [(0, 12, "FORM"), (23, 36, "FORM")]},
    {"entities": [(0, 12, "FORM"), (24, 37, "FORM")]},
    {"entities": [(0, 12, "FORM"), (26, 39, "FORM")]},
    {"entities": [(0, 12, "FORM"), (30, 43, "FORM")]},
    {"entities": [(0, 12, "FORM"), (24, 37, "FORM")]},
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
nlp.to_disk("form_model")


# Load the trained model
nlp = spacy.load("form_model")

# Define the text you want to evaluate
texts = [
    "Please enter final information into the form.",
    "Enter japanese comic book into the form.",
    "Could you fill in the form with Kyoto animation?",
    "Kindly provide New York view in the form.",
    "Make sure to put curry and rice in the form."
]

# Create a doc object for each text and print the entities
for text in texts:
    doc = nlp(text)
    print(f"Text: {text}")
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")
    print("\n")

"""

“Please enter [information] into the form.”
“Could you fill in the form with [information]?”
“Kindly provide your [information] in the form.”
“We need your [information] in the form.”
“Don’t forget to input your [information] in the form.”
“Make sure to put your [information] in the form.”

"""