# ch2-making-recommendations

To begin the tour of collective intelligence, I’m going to show you ways to use the
preferences of a group of people to make recommendations to other people. There
are many applications for this type of information, such as making product recom-
mendations for online shopping, suggesting interesting web sites, or helping people
find music and movies. This chapter shows you how to build a system for finding
people who share tastes and for making automatic recommendations based on things
that other people like.

You’ve probably come across recommendation engines before when using an online
shopping site like Amazon. Amazon tracks the purchasing habits of all its shoppers,
and when you log onto the site, it uses this information to suggest products you
might like. Amazon can even suggest movies you might like, even if you’ve only
bought books from it before. Some online concert ticket agencies will look at the his-
tory of shows you’ve seen before and alert you to upcoming shows that might be of
interest. Sites like reddit.com let you vote on links to other web sites and then use
your votes to suggest other links you might find interesting.

From these examples, you can see that preferences can be collected in many differ
ent ways. Sometimes the data are items that people have purchased, and opinions
about these items might be represented as yes/no votes or as ratings from one to five.
In this chapter, we’ll look at different ways of representing these cases so that they’ll
all work with the same set of algorithms, and we’ll create working examples with
movie critic scores and social bookmarking.

## Usage

have leiningen installed

`lein deps`

`lein test`