
# %% PROBLEM STATEMENT ---------------------------------------------------------
# Homework 1: Structured Data
#     Author: Leonard Armstrong
#        Log: 2021-Feb-04 (LA) Original version.
#    Problem: Performe data exploration and cleaning on a structured data set.
#             The Joebase action figure database is analyzed.


# %% LOAD LIBRARIES ------------------------------------------------------------

# Partial library imports
from mizani.formatters import dollar_format		# For formatting axes
from mizani.formatters import percent_format
from numpy import float64, int64

import matplotlib as mpl						# For figure resizing
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker				# For formatting plt axes
import pathlib as pl
import pandas as pd
import plotnine as gg							# "gg" refers to ggplot.
import squarify									# Generate treemaps.


# %% DEFINE GLOBAL CONSTANTS ---------------------------------------------------

DATA_PATH = "."									# Relative path to data file
DATA_FNAME = "action_figures_a.csv"				# Name of data file

# This map is used to rename a subset of columns. Not all columns are renamed.
COLUMN_RENAME_MAP = {
    "action_figure_description" : "af_descr",
	"FigureId"                  : "figure_id",
	"Manufacturer"              : "manufacturer",
	"Product"                   : "product_name",
	"ProductId"                 : "product_id",
    "Product Description"       : "product_descr",
	"Purchased From"            : "seller",
	"purchase_price"            : "price",
	"Release Year"              : "year",
}

# Define a set of one-hot-endcoded field for the genres.
GENRES = {
	"Adventure", "Air Force", "Armor", "Army", "Astronaut", "Avengers",
	"Celebrity", "Civilian", "Coast Guard", "Comics", "DC Comics", "Fashion",
	"Fire Fighter", "Foreign", "Horror", "Knight", "Marines", "Martial Arts", 
	"Marvel Comics", "Navy", "Police", "RAH/Cobra", "Sci-Fi", "Sports", "Spy", 
	"TV/Film", "Warrior", "Western", "World Leader", "X-Men" }

# Value to use when a year value is unknown
UNKNOWN_YEAR = 0

# Define a high-contrast color palette to assist those of us who are colorblind
# The palette is a subset of the colors defined in a palette published in
# https://jxnblk.com/blog/color-palette-documentation-for-living-style-guides/
MY_COLORS = [
	'#0074D9', '#2ECC40', '#FFDC00', '#FF4136', '#AAAAAA',
	'#DDDDDD', '#7FDBFF', '#39CCCC', '#3D9970', '#01ff70', '#FF851B' ]


# %% READ DATA -----------------------------------------------------------------

# Create a proper file path.
data_fpath = pl.Path(DATA_PATH, DATA_FNAME)
assert(data_fpath.exists())

# Read data into a pandas dataframe with default data types.
with open(data_fpath, "r") as datafile :
	fig_data = pd.read_csv(datafile, sep=',')
print(f'\nTHE DATA SET CONTAINS {fig_data.shape[0]} ROWS AND {fig_data.shape[1]} COLUMNS.\n')


# %% CLEAN THE DATA ------------------------------------------------------------

# Rename some of the columns, as desired.
fig_data.rename(columns=COLUMN_RENAME_MAP, inplace=True)

# Remove records with a null or NaN figure id. These represent figure costume
# sets, furniture, vehicles, etc, not figures.
non_figure_rows = fig_data[fig_data['figure_id'].isnull()].index
fig_data.drop(labels=non_figure_rows, axis=0, inplace=True)

# Set any unknown year to -1
fig_data["year"].fillna(UNKNOWN_YEAR, inplace=True)

# Update the data types as desired/required. Due to NA values these fields will
# be orginally read as float64. We can convert to int64 now that the offending
# NA values have been removed.
fig_data["figure_id"]=fig_data["figure_id"].astype('int64')
fig_data["year"]=fig_data["year"].astype('int64')

# Add a half-decade field as a string type. This works out better for graphing.
half_decade = fig_data["year"] - fig_data["year"].mod(5)
fig_data["Half Decade"] = [str(x) for x in half_decade]


# %% PROVIDE BASIC DATA SUMMARIES ----------------------------------------------

data_shape = fig_data.shape
print(f'\nTHE CLEANSED DATA SET CONTAINS {data_shape[0]} ROWS AND {data_shape[1]} COLUMNS.')

print('\nTHE DATA SET HAS THE FOLLOWING COLUMNS AND DATA TYPES:')
print(fig_data.dtypes)

### MANUFACTURERS AND SELLERS
print('\nTHE MANUFACTURER AND SELLER ATTRIBUTES HAS THE FOLLOWING STATISTCS:')
print(fig_data.describe(include='all')[{'manufacturer', 'seller'}])

### FIGURE PRICES
prices = fig_data[fig_data['year'] > 0]["price"].copy()
prices.dropna(inplace=True)

# Get the descriptive statistics for prices.
print('\nTHE PRICE ATTRIBUTE HAS THE FOLLOWING STATISTCS:')
print(prices.describe())

# Graph price distribution as a boxplot.
prices_df = pd.DataFrame({ 'x' : ['']*len(prices), 'price' : prices })
gg.options.figure_size=(4, 6)
g = (
	gg.ggplot(data=prices_df)
	+ gg.geom_boxplot(mapping=gg.aes(x='x', y='price'))
	+ gg.theme_bw()
	+ gg.ggtitle('Ranges of Prices Paid Across All Figures')
	+ gg.xlab('')
	+ gg.ylab('Price Paid')
	+ gg.scale_y_continuous(labels=dollar_format(digits=0)))
g.draw()
plt.show()

# Group figures by year and get counts per year.
year_gb = fig_data.groupby('year')
volume_per_year = year_gb.aggregate('count')
volume_per_year.drop(0, inplace=True)

# Plot a histogram of count of figures per year.
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 100 
fig, ax = plt.subplots()									# Create a graph
plt.bar(x=volume_per_year.index, height=volume_per_year["figure_id"])
ax.set_title('Year of Production of Figures in Collection')	# Title the chart
ax.set_xlabel('Year of Release')							# Title the x-axis
ax.set_ylabel('Volume of Figures')							# Title the y-axis
plt.show()
mpl.rcParams.update(mpl.rcParamsDefault)


# %% ---------------------------------------------------------------------------
# QUESTION 1: WHAT IS THE AVERAGE FIGURE PRICE PAID PER YEAR?

# First get a subset of figures with a valid year and price.
figs_with_prices = fig_data[(~fig_data["price"].isna()) & (fig_data["year"] > 0)]

# Group figures by year. Filter out figures with no price or an unknown year.
year_gb = figs_with_prices.groupby(by="year", axis=0)

# Get and print the inter-quartile ranges for each year.
iqr = (year_gb.quantile(q=0.75)['price']-year_gb.quantile(q=0.25)['price']).reset_index()
iqr.columns = ('Year', 'Price IQR')
print('\nTHE INTERQUARTILE RANGES FOR PRICE ARE:')
print(iqr)
iqr.to_csv('price_iqr.csv')

# Get the number of figures aquired from each production year.
num_aquired = year_gb.aggregate(len)["figure_id"]
avg_price = year_gb.aggregate('mean')["price"]

# Create a dataframe from the results
vol_price = pd.DataFrame({"volume" : num_aquired, "avg_price" : avg_price})

## Graph the results.

# First graph as a bar plot
# TODO: Change this to a ggplot with confidence interval.
fig, ax = plt.subplots()									# Create a graph
ax.plot(vol_price.index, vol_price['avg_price'], marker='x', linewidth=0.5)	# a bar graph
ax.set_title('Increase in Average Price Paid Per Figure')	# Title the chart
ax.set_ylabel('Average Paid for a Figure')					# Title the y-axis
ax.set_xlabel('Year of Acquisition')						# Title the x-axis
formatter = ticker.FormatStrFormatter('$%1.0f')				# Create a money formatter
ax.yaxis.set_major_formatter(formatter)						# Money-format the y-axis
plt.show()													# Dsplay the plot

# Then graph the price ranges as a boxplot
gg.options.figure_size=(14, 11)
g = (
	gg.ggplot(figs_with_prices)
	+ gg.geom_boxplot(mapping=gg.aes(x='factor(year)', y='price'))
	+ gg.theme_bw()
	+ gg.ggtitle('Ranges of Prices Paid for Figures Per Year')
	+ gg.xlab('Year of Acquisition')
	+ gg.ylab('Price Paid')
	+ gg.scale_y_continuous(labels=dollar_format(digits=0)))
g.draw()
plt.show()

# Sve figs_with_prices
figs_with_prices.to_csv('figs_with_prices.csv')

# %% ---------------------------------------------------------------------------
# QUESTION 2: HOW DID GENRE COLLECTING CHANGE OVER THE YEARS?

# Create volume summaries of the 30 genres
genre_volumes = pd.DataFrame(
	[ { "Genre" : g, "Volume" : sum(fig_data[g]) } for g in GENRES ] )
genre_volumes.sort_values('Volume', ascending=False, inplace=True)
print(genre_volumes)

# Set the figure's size and DPI.
mpl.rcParams['figure.figsize'] = [12.0, 9.0]
mpl.rcParams['figure.dpi'] = 100 

# Create a treemap
fig, ax = plt.subplots()
squarify.plot(
	sizes=genre_volumes["Volume"],
	label=genre_volumes['Genre'][:24], 
	color=MY_COLORS, alpha=0.7, ax=ax)
fig.suptitle('Tree Map of the Volume of Action Figures Within Genres',
	fontsize='x-large', fontweight='bold')
ax.set_title('Note: An action figure may be in more than one genre.')
plt.axis('off')
plt.tight_layout()
plt.show()

# Reset to default figure size
mpl.rcParams.update(mpl.rcParamsDefault)

# Get the sums of figures in all genres
genre_sums = fig_data.xs(key=GENRES, axis=1).sum()

# Get the genres with less than 10 figures
top_genres = set(genre_sums[genre_sums >= 35].index)

# Configure data for a relative percent bar plot.
year_gb = fig_data[fig_data["year"]>=1995].groupby(by="Half Decade", axis=0)
genre_yr_sums = year_gb.sum()

# Drop all columns except those needed for the plot and then reshape the data
# into a long form.
drop_cols = set(genre_yr_sums.columns) - top_genres
genre_yr_sums.drop(columns=drop_cols, inplace=True)
genre_yr_sums["Half Decade"] = genre_yr_sums.index
genre_df = genre_yr_sums.melt(id_vars=['Half Decade'])

# Generate the plot
g = (gg.ggplot(data=genre_df)
	+ gg.geom_col(
		mapping=gg.aes(x='Half Decade', y="value", fill="variable"),
		position=gg.position_fill())
	+ gg.ggtitle('Relative Percentages of Top Six Genres Over Time')
	+ gg.ylab('Relative Percent Across These Genres')
	+ gg.scale_fill_manual(values=MY_COLORS)
	+ gg.scale_y_continuous(labels=percent_format()))
g.draw()
plt.show()

# Save genre_df
genre_df.to_csv('genre_df.csv')


# %% ---------------------------------------------------------------------------

# QED!