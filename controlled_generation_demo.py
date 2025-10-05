#!/usr/bin/env python3
"""
Simple demonstration of controlled text generation
"""

from markov_text_generator import MarkovTextGenerator

# Example 1: Coherent text input
coherent_text = """
The sun rises in the east every morning. Every morning brings new opportunities and fresh beginnings. Fresh beginnings allow us to start over and make positive changes. Positive changes lead to personal growth and development. Personal development is essential for achieving our goals and dreams.

Learning new skills is important for career advancement. Career advancement requires dedication and continuous improvement. Continuous improvement helps us stay competitive in the job market. The job market is constantly evolving with new technologies. New technologies create exciting possibilities for innovation and creativity.

Innovation drives progress in science and technology. Technology has revolutionized the way we communicate and work. Work-life balance is crucial for maintaining good health and happiness. Happiness comes from meaningful relationships and personal fulfillment. Personal fulfillment is achieved through pursuing our passions and helping others.
"""

# Example 2: Story-like text
story_text = """
Once upon a time there was a brave knight. The brave knight lived in a magnificent castle. The magnificent castle stood on top of a tall mountain. The tall mountain overlooked a peaceful valley. The peaceful valley was home to many friendly villagers.

The friendly villagers worked hard every day. Every day they tended to their gardens and farms. Their gardens were filled with beautiful flowers and vegetables. The vegetables grew strong and healthy in the fertile soil. The fertile soil was blessed by the morning rain and sunshine.

One day the knight decided to explore the enchanted forest. The enchanted forest was full of magical creatures and mysterious sounds. The mysterious sounds echoed through the ancient trees. The ancient trees had stood for hundreds of years. Hundreds of years of history were carved into their mighty trunks.
"""

def demo_generation():
    print("=== Controlled Text Generation Demo ===\n")
    
    # Test with coherent text
    print("1. Training with coherent, educational text:")
    print("-" * 50)
    
    generator1 = MarkovTextGenerator()
    generator1.train_from_text(coherent_text)
    
    for i in range(3):
        result = generator1.generate_text(25)
        print(f"Generation {i+1}: {result}")
        print()
    
    print("\n" + "="*60 + "\n")
    
    # Test with story text
    print("2. Training with story-like text:")
    print("-" * 50)
    
    generator2 = MarkovTextGenerator()
    generator2.train_from_text(story_text)
    
    for i in range(3):
        result = generator2.generate_text(25)
        print(f"Generation {i+1}: {result}")
        print()

if __name__ == "__main__":
    demo_generation()