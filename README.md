# Where Is My Book?

## 1. Motivation

"Where is my (disired) book?" is my question whenever I come to Fulbright University Vietnam (FUV) library. Our school library has a quiet impression numbers of books, and since we do not use any digital techniques to manage, it is tedious for me to skim to all book spines and find where is the book I want to borrow. 

Then, as part of learning journey in the course ENG 301 - Computer Vision instructed by Dr. Duong Phung, I have this chance to make use of the foundation of Computer Vision, Image Processing, and Algorithms as an attempt of building an application that can help me and other Fulbright students to spot their wanted books.

## 2. Assumptions 

Acknowledging there is a big gap between my naive algorithm and the complexity of the problem itself, I would like to introduce some assumptions that could help me to simplify the problem. Please note that, without these assumptions, the program may be crashes due to the lack of essential information

- Due to the exclusiveness, the program is not expected to work well with bookshelves that not in Fulbright. In details, we have a blur circle stick on each book spine and this program making use of this observation for book spine segmentation.
- For the lastest version, the program can handle images containing a single line of bookshelf only. Multiple lines of book is not yet implemented.
