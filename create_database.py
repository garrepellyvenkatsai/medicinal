import sqlite3

# Database file path
DATABASE_PATH = 'plants.db'

# Connect to SQLite database (it will create the file if it doesn't exist)
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# Create the `plant_care` table
cursor.execute('''
CREATE TABLE IF NOT EXISTS plant_care (
    plant_name TEXT PRIMARY KEY,
    sunlight_requirements TEXT,
    watering_schedule TEXT,
    soil_type TEXT,
    additional_tips TEXT
);
''')

# Create the `recipes` table
cursor.execute('''
CREATE TABLE IF NOT EXISTS recipes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_name TEXT,
    recipe_name TEXT,
    instructions TEXT,
    FOREIGN KEY (plant_name) REFERENCES plant_care(plant_name)
);
''')

# Create the `symptoms_to_plants` table
cursor.execute('''
CREATE TABLE IF NOT EXISTS symptoms_to_plants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symptom TEXT,
    plant_name TEXT,
    FOREIGN KEY (plant_name) REFERENCES plant_care(plant_name)
);
''')

# Insert sample data
cursor.execute('''
INSERT OR IGNORE INTO plant_care (plant_name, sunlight_requirements, watering_schedule, soil_type, additional_tips)
VALUES 
    ('Amla', 'Full sunlight', 'Water regularly', 'Well-drained soil', 'Prune to maintain size.'),
    ('Amruthaballi', 'Partial sunlight', 'Water moderately', 'Loamy soil', 'Provide support for climbing.'),
    ('Arali', 'Full sunlight', 'Water occasionally', 'Sandy or loamy soil', 'Avoid waterlogging.'),
    ('Ashoka', 'Partial sunlight', 'Water weekly', 'Rich, well-drained soil', 'Mulch to retain moisture.'),
    ('Asthma_weed', 'Full sunlight', 'Water sparingly', 'Dry, sandy soil', 'Grows wild, manage spread.'),
    ('Badipala', 'Partial sunlight', 'Keep soil moist', 'Loamy soil', 'Needs high humidity.'),
    ('Balloon_Vine', 'Full sunlight', 'Water moderately', 'Well-drained soil', 'Provide trellis for support.'),
    ('Bamboo', 'Full sunlight', 'Water daily', 'Rich, moist soil', 'Ensure good drainage.'),
    ('Beans', 'Full sunlight', 'Water regularly', 'Fertile, well-drained soil', 'Use stakes for climbing types.'),
    ('Betel', 'Partial sunlight', 'Keep soil moist', 'Loamy, rich soil', 'Avoid direct harsh sunlight.'),
    ('Bhrami', 'Partial sunlight', 'Water frequently', 'Moist, clayey soil', 'Great for ground cover.'),
    ('Bringaraja', 'Partial sunlight', 'Water regularly', 'Loamy soil', 'Harvest leaves often.'),
    ('Camphor', 'Full sunlight', 'Water occasionally', 'Well-drained soil', 'Avoid frost exposure.'),
    ('Caricature', 'Partial sunlight', 'Water lightly', 'Sandy soil', 'Can be grown indoors.'),
    ('Castor', 'Full sunlight', 'Water occasionally', 'Well-drained soil', 'Tolerates drought conditions.'),
    ('Catharanthus', 'Full sunlight', 'Water sparingly', 'Well-drained soil', 'Prune to encourage blooms.'),
    ('Chakte', 'Partial sunlight', 'Keep soil moist', 'Clay or sandy soil', 'Protect from pests.'),
    ('Chilly', 'Full sunlight', 'Water moderately', 'Loamy soil', 'Avoid wetting leaves.'),
    ('Citron lime (herelikai)', 'Full sunlight', 'Water deeply once a week', 'Well-drained soil', 'Fertilize during growth.'),
    ('Coffee', 'Partial sunlight', 'Water regularly', 'Rich, acidic soil', 'Shade is preferred.'),
    ('Common rue (naagdalli)', 'Full sunlight', 'Water sparingly', 'Well-drained soil', 'Prune to maintain shape.'),
    ('Coriander', 'Partial sunlight', 'Water lightly', 'Moist soil', 'Harvest leaves regularly.'),
    ('Curry', 'Full sunlight', 'Water weekly', 'Loamy, well-drained soil', 'Pinch tips for bushy growth.'),
    ('Doddpathre', 'Partial sunlight', 'Keep soil moist', 'Loamy soil', 'Avoid overwatering.'),
    ('Drumstick', 'Full sunlight', 'Water occasionally', 'Well-drained sandy soil', 'Tolerates drought well.'),
    ('Ekka', 'Partial sunlight', 'Keep soil moist', 'Rich, well-drained soil', 'Ensure humidity levels.');

''')

cursor.execute('''
INSERT OR IGNORE INTO recipes (plant_name, recipe_name, instructions)
VALUES 
    ('Amla', 'Amla Juice', 'Blend fresh amla. add honey and water. and strain before drinking.'),
    ('Amruthaballi', 'Herbal Infusion', 'Boil leaves in water and strain to make a tea.'),
    ('Arali', 'Arali Paste', 'Crush fresh leaves and apply externally for skin conditions.'),
    ('Ashoka', 'Ashoka Bark Tea', 'Steep dried bark in boiling water for 10 minutes.'),
    ('Asthma_weed', 'Asthma Decoction', 'Boil leaves with honey and consume as needed.'),
    ('Badipala', 'Herbal Oil', 'Infuse leaves in coconut oil and use for skin issues.'),
    ('Balloon_Vine', 'Balloon Vine Soup', 'Cook leaves with spices and strain for soup.'),
    ('Bamboo', 'Bamboo Shoot Curry', 'Boil shoots, sauté with spices, and serve.'),
    ('Beans', 'Steamed Beans', 'Steam beans and season with salt and pepper.'),
    ('Betel', 'Betel Leaf Digestive', 'Wrap areca nut and spices in a leaf and chew.'),
    ('Bhrami', 'Bhrami Chutney', 'Blend leaves with green chilies, tamarind, and salt.'),
    ('Bringaraja', 'Hair Oil', 'Boil leaves in coconut oil and apply to scalp.'),
    ('Camphor', 'Camphor Balm', 'Mix camphor powder with coconut oil for topical use.'),
    ('Caricature', 'Leaf Infusion', 'Steep leaves in hot water and strain.'),
    ('Castor', 'Castor Oil Preparation', 'Extract oil from seeds and use externally.'),
    ('Catharanthus', 'Catharanthus Tea', 'Steep flowers in boiling water and strain.'),
    ('Chakte', 'Chakte Paste', 'Grind leaves and apply to wounds.'),
    ('Chilly', 'Chilly Sauce', 'Blend chilies with vinegar, salt, and garlic.'),
    ('Citron lime (herelikai)', 'Citron Juice', 'Squeeze juice from fruits, add sugar and water.'),
    ('Coffee', 'Fresh Brew', 'Grind coffee beans, brew with hot water, and strain.'),
    ('Common rue (naagdalli)', 'Rue Infusion', 'Steep leaves in hot water for 5 minutes.'),
    ('Coriander', 'Coriander Chutney', 'Grind leaves with coconut, chilies, and tamarind.'),
    ('Curry', 'Curry Leaf Powder', 'Dry leaves. grind to powder, and use in recipes.'),
    ('Doddpathre', 'Doddpathre Tea', 'Boil leaves with jaggery and consume warm.'),
    ('Drumstick', 'Drumstick Soup', 'Boil pods, blend with spices, and serve warm.'),
    ('Ekka', 'Herbal Ekka Paste', 'Grind leaves and apply for skin issues.');

''')

cursor.execute('''
INSERT OR IGNORE INTO symptoms_to_plants (symptom, plant_name)
VALUES 
    ('Immunity Boosting', 'Amla'),
    ('Cough and Cold', 'Amruthaballi'),
    ('Skin Infections', 'Arali'),
    ('Menstrual Pain', 'Ashoka'),
    ('Respiratory Issues', 'Asthma_weed'),
    ('Skin Rashes', 'Badipala'),
    ('Joint Pain', 'Balloon_Vine'),
    ('Bone Strength', 'Bamboo'),
    ('Nutritional Deficiency', 'Beans'),
    ('Digestive Aid', 'Betel'),
    ('Memory Enhancement', 'Bhrami'),
    ('Hair Health', 'Bringaraja'),
    ('Aromatic Relief', 'Camphor'),
    ('Minor Cuts', 'Caricature'),
    ('Constipation', 'Castor'),
    ('Diabetes Support', 'Catharanthus'),
    ('Wound Healing', 'Chakte'),
    ('Metabolism Boost', 'Chilly'),
    ('Vitamin C Deficiency', 'Citron lime (herelikai)'),
    ('Energy Boost', 'Coffee'),
    ('Digestive Support', 'Common rue (naagdalli)'),
    ('Flavor Enhancer', 'Coriander'),
    ('Appetite Stimulation', 'Curry'),
    ('Respiratory Relief', 'Doddpathre'),
    ('Iron Deficiency', 'Drumstick'),
    ('Skin Irritation', 'Ekka');
''')

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database created and initialized successfully.")
