
import json

def glove_text_to_json(glove):

    # Initialize an empty list to store the JSON instances
    json_data = {}

    # Read the input text file
    with open(glove, 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line into words
            words = line.split()

            # Extract the first element as the value
            value = words[0]

            # Convert the remaining elements into a list
            keys = words[1:]

            # Create a JSON instance
            json_data[value] = keys
                

    # Write the JSON data to an output JSON file
    with open('glove_200d.json', 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=1)

    print("Conversion complete.")

glove_text_to_json('glove.twitter.27B/glove.twitter.27B.200d.txt')