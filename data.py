dataset = [
    "You have the right to perform your prescribed duty, but you are not entitled to the fruits of your actions.",
    "There is neither this world nor the world beyond. Neither happiness for the one who doubts nor liberation for the one who does not know.",
    "One who has control over the mind is tranquil in heat and cold, in pleasure and pain, and in honor and dishonor; and is ever steadfast with the Supreme Self.",
    "A person can rise through the efforts of his own mind; mind is the friend and mind is the enemy of each individual.",
    "A man's own self is his friend. A man's own self is his foe.",
    "The soul is neither born, and nor does it die.",
    "There is neither this world, nor the world beyond. Nor happiness for the one who doubts.",
    "The power of God is with you at all times; through the activities of mind, senses, breathing, and emotions; and is constantly doing all the work using you as a mere instrument.",
    "Set thy heart upon thy work, but never on its reward.",
    "When meditation is mastered, the mind is unwavering like the flame of a lamp in a windless place."
]

# Save the dataset to a text file
with open("bhagavad_gita_quotes.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(dataset))
