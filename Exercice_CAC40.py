import csv

def open_csv(file):
    
    with open(file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    
    return data

def parse_data(data):
    parsed_data = []
    
    for row in data:
        parsed_row = {
            'date': row['Date'],
            'ouv': row['Ouverture'],
            'haut': row['Haut'],
            'bas': float(row['Market Cap'].replace(',', '').replace('$', '')),
            'Price': float(row['Price'].replace(',', '').replace('$', '')),
            'PE Ratio': float(row['PE Ratio']) if row['PE Ratio'] != 'N/A' else None
        }
        parsed_data.append(parsed_row)
    
    return parsed_data

print(open_csv('./CAC40_2026-01-21.csv'))