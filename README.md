# Overview
This repo contains a solution for [problem_stores](https://gist.github.com/alejandrofsil/e3142b28d34ece49be5ef31462908b9b). 

# Problem

At Zara we want to open 15 new physical stores in Germany.

As a member of the Data & Analytics team, you are assigned this research. You have to analyze the problem, come to a solution and propose 15 new physical stores.

# Assignment

Given the sales and the current open stores in Germany, propose 15 new locations following these rules:

- New stores _must_ be located on zip codes that already have eCommerce sales.
- All stores (new and existing) _must_ be at least 20 km apart from each other.
- The algorithm _must_ maximize the online sales within a 10 km radius around the newly proposed stores.

Use [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula) to get the distance between two points.

# Solution

Given the rules, from my point of view the best solution is straightforward and simple:
Propose stores sequentially, starting from the one with the highest sum of zipcode units in 10km or less with two constraints: that the new store is 20km or more from another store already built (or previously proposed) and that zipcodes already covered by a store are not taken into account for the computation of new stores. The following stores are computed in the same way but adding the stores predicted above to the list of current stores

Details about implementation:
 - It is important to avoid calculating distances already made in previous iterations.
 - It is important to use sparse matrix (scipy) for the distance matrix which contains a large number of zeros because most zip codes do not contribute to the sales of a location.
This algorithm takes approximately 2 seconds
![image]()
