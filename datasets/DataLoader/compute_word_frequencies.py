
from collections import Counter
import re

# Function to read the text file and compute word frequencies
def compute_word_frequencies(filename, blacklist, top_n=30):
    word_counter = Counter()
    
    # Read the file and update the word frequencies
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Using regex to find words and ignore punctuation
            words = re.findall(r'\b\w+\b', line.lower())
            # Remove blacklisted words
            words = [word for word in words if word not in blacklist]
            word_counter.update(words)
    
    print("Unique words:", len(word_counter))

    print("Words total ",word_counter.total())

    # Get the top N most frequent words
    most_common_words = word_counter.most_common(top_n)
    
    return most_common_words


def print_word_frequencies(word_freq):
    #words, frequencies = zip(*word_freq)
    print(word_freq)



# Function to plot the word frequencies
def plot_word_frequencies(word_freq):
    import matplotlib.pyplot as plt

    words, frequencies = zip(*word_freq)
    
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {len(words)} Most Frequent Words')
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Path to the text file
    filename = 'all_labels.txt'
    
    # Compute word frequencies (set top_n based on visibility needs)
    top_n = 500  # Change this number to adjust the number of top words shown
    
    blacklist     = ["(", ")", ",", ".", "a", "an", 's', 'hu', 'hy',  'w']

    # List of blacklisted words
    stopwords = [ "of", "on", "and", "I", "in", "the", "is", "it", "at", "to", "with", "for", "from", "near", "while"]
    blacklist.extend(stopwords)

    overrepresented = ['top','next','two', 'are', 'it', 'its', 'up', 'down', 'left', 'right', 'in', 'out', 'front', 'to', 'has', 'by']
    blacklist.extend(overrepresented)

    difficult = ['nintendo', 'wii',  'umpire']
    blacklist.extend(difficult)
     
    #verbs = ['abandoned', 'adjusting', 'advertising', 'alcoholic', 'allowed', 'arm', 'arranged', 'arrive', 'assorted', 'attached', 'bags', 'baked', 'bakery', 'baking', 'barn', 'bathing', 'batter', 'batting', 'beach', 'beak', 'bear', 'believing', 'benches', 'bending', 'bicycles', 'billowing', 'biplane', 'biscuit', 'biting', 'blanket', 'blender', 'block', 'blooming', 'blow', 'blowing', 'bookcase', 'bookstore', 'booze', 'bottle', 'boulders', 'bow', 'broken', 'brushing', 'bubbles', 'buffalo', 'building', 'bun', 'bundt', 'burning', 'camping', 'capped', 'carpeted', 'carriage', 'carrying', 'catch', 'catching', 'cauliflower', 'cave', 'celebrating', 'cellar', 'chained', 'chasing', 'checkered', 'checking', 'chewing', 'choir', 'chopped', 'clad', 'claim', 'cleaning', 'climbing', 'close', 'closed', 'colored', 'combing', 'coming', 'competing', 'connected', 'consisting', 'containing', 'controls', 'conveyor', 'cooked', 'cooking', 'cooling', 'counter', 'cover', 'covered', 'cross', 'crossed', 'crossing', 'crowd', 'crowded', 'curtains', 'cut', 'cute', 'cutting', 'dance', 'dancing', 'decorated', 'decorating', 'decorative', 'depicting', 'desserts', 'digging', 'dining', 'dipping', 'directing', 'displayed', 'do', 'dock', 'docked', 'doing', 'dolphin', 'double', 'doubles', 'drawing', 'drawn', 'dressed', 'dressing', 'dribbling', 'dried', 'drink', 'drinking', 'driven', 'driving', 'drum', 'drying', 'duckling', 'dump', 'eagle', 'eat', 'eaten', 'eating', 'exhibit', 'expired', 'face', 'fallen', 'falling', 'fashioned', 'fast', 'feeding', 'feeds', 'fenced', 'fighting', 'filled', 'fishing', 'fix', 'fixing', 'flavored', 'floating', 'flooded', 'flooring', 'flowing', 'flown', 'fly', 'flying', 'forested', 'fried', 'frosted', 'frosting', 'fry', 'frying', 'galloping', 'gathered', 'gathering', 'gear', 'get', 'getting', 'glassware', 'glazed', 'glove', 'glowing', 'go', 'going', 'goose', 'grab', 'grasses', 'grazing', 'grinding', 'growing', 'hand', 'handle', 'handwritten', 'hanging', 'has', 'hate', 'having', 'headphones', 'held', 'helmet', 'helping', 'herded', 'herding', 'hiking', 'hit', 'hitting', 'hold', 'holding', 'holds', 'hooded', 'hooked', 'horned', 'horseback', 'hugging', 'icing', 'includes', 'including', 'jeep', 'jump', 'jumping', 'kick', 'kicking', 'kind', 'kissing', 'kiteboarding', 'kites', 'kitten', 'kneeling', 'knick', 'laid', 'landing', 'laying', 'leading', 'leads', 'leaf', 'leafless', 'leaning', 'leash', 'leaves', 'left', 'lettering', 'lettuce', 'lift', 'lighting', 'lined', 'link', 'lit', 'living', 'loaded', 'loading', 'lobby', 'lodge', 'log', 'look', 'looking', 'looks', 'lounging', 'lush', 'lying', 'made', 'make', 'making', 'marching', 'match', 'measuring', 'merry', 'microwave', 'microwaves', 'mid', 'milking', 'miss', 'mixed', 'mixing', 'monitor', 'moped', 'motes', 'motorized', 'mound', 'mounted', 'mouse', 'nightstands', 'note', 'nurses', 'objects', 'open', 'opening', 'ornate', 'outfit', 'outfits', 'outstretched', 'oven', 'overlooking', 'pack', 'packed', 'paddling', 'paintbrush', 'painted', 'painting', 'pair', 'pairs', 'pan', 'paneling', 'parachute', 'parasail', 'parasailing', 'parked', 'passes', 'passing', 'patch', 'patterned', 'pay', 'pears', 'peeled', 'pepperoni', 'perched', 'perform', 'performing', 'petting', 'photograph', 'pick', 'piled', 'pillows', 'pitching', 'play', 'played', 'playground', 'playing', 'plays', 'podium', 'pointing', 'pose', 'poses', 'posing', 'poured', 'pouring', 'practicing', 'prepared', 'preparing', 'produce', 'propped', 'pull', 'pulled', 'pulling', 'pump', 'pushing', 'putting', 'racing', 'rack', 'racquet', 'radishes', 'railing', 'rainbow', 'ram', 'reading', 'reads', 'rear', 'reflected', 'reflecting', 'relish', 'remote', 'remove', 'ridden', 'ride', 'rides', 'riding', 'rink', 'ripe', 'roast', 'rocking', 'roll', 'roofed', 'rose', 'row', 'rowing', 'rug', 'run', 'running', 'saddle', 'sail', 'sailing', 'salad', 'salads', 'sand', 'sandbags', 'sauce', 'says', 'seagulls', 'seen', 'selling', 'served', 'set', 'setting', 'sewing', 'shaking', 'shaped', 'sheared', 'shining', 'shorts', 'shot', 'shoveling', 'show', 'shower', 'showing', 'shown', 'sign', 'singing', 'sink', 'sit', 'sits', 'sitting', 'skate', 'skateboarding', 'skating', 'ski', 'skiing', 'skirt', 'sled', 'sleeping', 'slice', 'sliced', 'sliding', 'smile', 'smiling', 'smoke', 'smoking', 'smoothie', 'sniffing', 'snowboarding', 'snowmobile', 'soaked', 'soda', 'sold', 'spanning', 'speaking', 'spewing', 'split', 'spoon', 'spooning', 'spraying', 'sprinkles', 'squatting', 'stacked', 'stadium', 'stained', 'staircase', 'stand', 'standing', 'stands', 'staring', 'steam', 'steering', 'stem', 'stew', 'stick', 'sticking', 'stir', 'stirred', 'stirring', 'stop', 'stopped', 'store', 'stove', 'strawberry', 'stream', 'stripe', 'striped', 'stuffed', 'surfer', 'surfing', 'surrounded', 'swim', 'swimming', 'swims', 'swimsuit', 'swimsuits', 'swinging', 'syrup', 'tag', 'take', 'taken', 'taking', 'talk', 'talking', 'tasting', 'teaching', 'tending', 'throw', 'throwing', 'tie', 'tied', 'tiled', 'toaster', 'tomatoes', 'tongs', 'tooth', 'toothbrush', 'topped', 'tops', 'tossing', 'touch', 'touching', 'tour', 'tow', 'towel', 'towering', 'track', 'train', 'traveling', 'trekking', 'trick', 'trying', 'tying', 'typing', 'uniformed', 'unmade', 'use', 'used', 'using', 'vandalized', 'vending', 'waiting', 'wake', 'walk', 'walking', 'walled', 'washing', 'watch', 'watched', 'watches', 'watching', 'watering', 'wearing', 'wetsuit', 'wheelbarrow', 'wheeled', 'whipped', 'windsurfing', 'wooded', 'worked', 'working', 'worn', 'writing', 'yard']
    #blacklist.extend(verbs)

    word_freq = compute_word_frequencies(filename, blacklist, top_n)
    
    # Sort by frequency (already sorted by most_common)
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
    
    print_word_frequencies(word_freq)

    # Plot the word frequencies
    plot_word_frequencies(word_freq)

if __name__ == '__main__':
    main()





