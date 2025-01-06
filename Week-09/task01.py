import re

def main():
    test_string = "Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"

    print(f'The sentences: {re.split(r'(?<=[?!.])\s+', test_string)}')
    print(f'All capitalized words: {re.findall(r'[A-Z][a-z]*', test_string)}')
    print(f'The string split on spaces: {re.split(r'\s+', test_string)}')
    print(f'All numbers: {re.findall(r'\d+', test_string)}')

if __name__ == '__main__':
    main()