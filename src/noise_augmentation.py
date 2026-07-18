import random


# -------------------------------
# Word Deletion
# -------------------------------
def delete_random_word(sentence):

    words = sentence.split()

    if len(words) <= 1:
        return sentence

    index = random.randint(0, len(words) - 1)

    del words[index]

    return " ".join(words)


# -------------------------------
# Character Deletion
# -------------------------------
def delete_random_character(sentence):

    if len(sentence) <= 3:
        return sentence

    index = random.randint(0, len(sentence) - 1)

    return sentence[:index] + sentence[index + 1:]


# -------------------------------
# Character Insertion
# -------------------------------
def insert_random_character(sentence):

    letters = "abcdefghijklmnopqrstuvwxyz"

    index = random.randint(0, len(sentence))

    random_letter = random.choice(letters)

    return sentence[:index] + random_letter + sentence[index:]


# -------------------------------
# Character Swap
# -------------------------------
def swap_adjacent_characters(sentence):

    if len(sentence) < 2:
        return sentence

    index = random.randint(0, len(sentence) - 2)

    sentence = list(sentence)

    sentence[index], sentence[index + 1] = (
        sentence[index + 1],
        sentence[index]
    )

    return "".join(sentence)


# -------------------------------
# Word Duplication
# -------------------------------
def duplicate_random_word(sentence):

    words = sentence.split()

    if len(words) == 0:
        return sentence

    index = random.randint(0, len(words) - 1)

    words.insert(index, words[index])

    return " ".join(words)


# -------------------------------
# Random Noise Generator
# -------------------------------
def add_asr_noise(sentence):

    operations = [

        delete_random_word,
        delete_random_character,
        insert_random_character,
        swap_adjacent_characters,
        duplicate_random_word

    ]

    operation = random.choice(operations)

    return operation(sentence)


# -------------------------------
# Demo
# -------------------------------
if __name__ == "__main__":

    random.seed(42)

    sentences = [

        "play the music",
        "increase the volume",
        "pause the song",
        "pick up the call"

    ]

    print("=" * 60)
    print("ASR Noise Augmentation Demo")
    print("=" * 60)

    for sentence in sentences:

        noisy = add_asr_noise(sentence)

        print(f"\nOriginal : {sentence}")
        print(f"Noisy    : {noisy}")