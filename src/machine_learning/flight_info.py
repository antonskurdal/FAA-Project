from pathlib import Path
import pandas as pd

# Set up directory
parent_directory = Path("D:/#FAA UAS Project/OpenSky WEEK/Individual Aircraft/batch_3/output")
file = parent_directory / "agg_3.csv"
df = pd.read_csv(file)

""" # Drop erroneous
df = df[df['taxonomy'] != 'erroneous'] """

""" # Drop noise
df = df[df['taxonomy'] != 'noise'] """

print("Taxonomy Counts:\n{}\n".format(df['taxonomy'].value_counts()))
print("Total Points: {}".format(df.shape[0]))
print("Unique Aircraft: {}".format(df['icao24'].unique().size))