# [CNN for postprocessing of IVT images]
<img align="right" src="static/favicon.ico" alt="logo" width="220">

*Implementation of a Convolutional Neural Network (CNN) for the postprocessing of Integrated water Vapor Transport a tourist office of a cultural tourism destination for the Hypermedia Applications course of Politecnico di Milano.*

## Authors
[Alessandro Austoni](https://github.com/AlessandroAustoni) - cod:10524152 (alessandro.austoni@mail.polimi.it) - Geoinformatics Engineering<br/>
- Design Document, Supervision of Design and Implementation consistency, CSS Styling improvement  

[Matteo Beltrante](https://github.com/Beltrante) - cod: 10575137 (matteo.beltrante@mail.polimi.it) - Computer Science<br>
- Backend (Api endpoints and Database management), Dynamic Data handling, Website structure 

[Francesco Bosso](https://github.com/FBosso) - cod: 10555965 (francesco.bosso@mail.polimi.it) - Geoinformatics Engineering<br>
- Frontend (Pages, Layout and Components structure), Graphic Design, SEO optimization

[Thomas Martinoli](https://github.com/ThomasMartinoli) - cod: 10502028 (thomas.martinoli@mail.polimi.it) - Geoinformatics Engineering<br>
- Breadcrumbs implementation, Data Gathering, Interaction scenarios


## Project Description
The project consists in the design and the implementation of a website for a tourist office of a cultural tourism destination. The requirements contain some constraints about the content and relationships of the main entities of the website.
The project is divided into three main parts: Design, Backend, Frontend. 

## Design
Design Document contains the final output of our design process and contain the following sections:
1. Table of Contents
2. C-IDM Diagram
3. Content-in-the small Tables
4. Abstract Pages 
5. Commented Wireframes
6. Interaction Scenarios</br>

The full design is available here [here](insert link here).

## Backend Technologies
+ [NPM](https://docs.npmjs.com/) is the Node Package Manager used in this project to download and manage the required modules.
+ [NodeJS](https://nodejs.org/it/docs/) is used in the implementation to create a fully working Web Server by using the Express module.
+ [Express](https://expressjs.com/it/) is a module providing a flexible Node.js web application framework that provides a robust set of features to develop web and mobile applications.
+ [PostgreSQL](https://www.postgresql.org/docs/) is an open source DB used to store the data required to populate the HTML pages
+ [Sequelize](https://sequelize.org/) is a promise-based Node.js Object–relational mapping tool used to make querying the database during API calls easier 

## Frontend Technologies
+ [Axios](https://axios-http.com/docs/intro) is promise-based HTTP Client for node.js and is used to handle all the requests to the server's API
+ [HTML](https://developer.mozilla.org/en-US/docs/Web/HTML?retiredLocale=it)/[CSS](https://developer.mozilla.org/en-US/docs/Web/CSS?retiredLocale=it) are used to structure and style the different pages in the website 

## Framework and Key Ideas 
[NuxtJS](https://nuxtjs.org/) is the framework we used, which allows to create web application utilizing the Vue.js single file component system, allowing for, among others, reduced time to interactivity and improved SEO.
Nuxt also offers special directories to handle different features, in this project we utilized:
+ The **pages** directory to store our pages, and organize the routing structure.
+ The **components** directory to store our reusable components used in each page.
+ The **layouts** directory to define the default page layout with a header and a footer and the error page layout.
<br> The Header and the Footer contain a list of landmarks with access to all the main pages of the site, the footer also has social media contacts and a small snippet about locus vision.
+ The **server** directory to save our backend implementation 
+ We also made extensive use of the routing nuxt provide, with the addition of index and underscored vue pages in key pages sub directories in order to obtain more readable urls for the user.</br>

Thanks to the use of this functionalities we believe we were able to stay true to the core concepts behind Vue and Nuxt developement and mantain our website source code as simple and easily scalable as possible 

## Components Used 
During this project developement we tried to make as much use of components as possible, with code reusability and possible future additions in mind, we settled with the use of 17 different components which are used by multiple different pages, three of them, like the [TopMapDescription](https://github.com/FBosso/Locus/blob/main/components/TopMapDescription.vue), are an aggregate of smaller ones and were included after careful considerations **exclusively** to improve code readability.
<br>This *aggregate* components are:
  [TopEventDescription](https://github.com/FBosso/Locus/blob/main/components/TopEventDescription.vue)
, [TopMapDescription](https://github.com/FBosso/Locus/blob/main/components/TopMapDescription.vue)
, [TopSegmentDescription](https://github.com/FBosso/Locus/blob/main/components/TopSegmentDescription.vue)
<br>
Each component code is also commented with a short description of their functions and a full list is available [here](https://github.com/FBosso/Locus/tree/main/components)   

## Deploy Decisions  
Finally we decided to deploy our project as a **Static website** using GitHub pages based on the idea that our Art Town's city council would be able to plan its Attractions in advance, allowing the site to be updated no more frequently than every 6 months.
The Static deploy was chosen to guarantee the following benefits:
+ The site does not need a running backend with its corresponding costs like in Server Side rendering 
+ The APIs are only called during the static generation of the site, thus no malicious  or malformed request needs to be accounted for, making the site more secure.   
+ Having the pages already populated with data allow for a better SEO rating than Client Side rendering
+ Pages are loaded faster resulting in a better browsing experience for the user

  
## Build Setup

```bash
# install dependencies
$ npm install

# serve with hot reload at localhost:3000
$ npm run dev

# build for production and launch server
$ npm run build
$ npm run start

# generate static project (for Nuxt2 run dev on another terminal)
$ npm run generate
```
For detailed explanation on how things work, check out the [documentation](https://nuxtjs.org).
