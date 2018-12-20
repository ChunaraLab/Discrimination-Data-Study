rm(list=ls())  
ls() 

# set directory
setwd( "/Users/kunalrelia/Desktop/Hate Crime data/Code/graphs/USMap")

#load libraries
library("ggplot2")
library(ggrepel)
library("maps")

#load racism and hate crime data and format city names
racism<-read.csv("racism.csv")
racism<-racism[1:100,2:11]
racism$City<-as.character(racism$City)
racism$State.Abbreviation<-as.character(racism$State.Abbreviation)
racism$State<-as.character(racism$State)
racism$Total.number.of.Tweets.made.over.6.years<-as.numeric(racism$Total.number.of.Tweets.made.over.6.years)
racism$Prevalence.of.Racism.in....x.10..2.<-as.numeric(racism$Prevalence.of.Racism.in....x.10..2.)
racism$City[13]<-"Las Vegas"
racism$City[19]<-"Charlotte"
racism$City[21]<-"Lakewood"
racism$City[26]<-"Saint Paul"
racism$City[36]<-"Neptune City"

#sort the data in ascending order based on racism hate crimes 
racism<-racism[order(racism$Total.RHC...Numbers.across.6.years),]

#find the average number of racism hate crimes in each city
racism$Total.RHC...Numbers.across.6.years<-racism$Total.RHC...Numbers.across.6.years/6

#Add racism hate crime categories
racism$HC<-"Medium HC"
racism$HC[racism$Total.RHC...Numbers.across.6.years<=4]<-"Low HC"
racism$HC[racism$Total.RHC...Numbers.across.6.years>9]<-"High HC"

#load US city latitude and longitude data
cities1<-read.csv("USCities.csv")
racism$lon<-0
racism$lat<-0

cities1$Country<-as.character(cities1$Country)
cities1$City<-as.character(cities1$City)
cities1$AccentCity<-as.character(cities1$AccentCity)
cities1$Region<-as.character(cities1$Region)
cities1$Population<-as.numeric(cities1$Population)
cities1$Latitude<-as.numeric(cities1$Latitude)
cities1$Longitude<-as.numeric(cities1$Longitude)

#map latitude and longitude to cities
for(i in 1:nrow(racism)){
  city<-racism$City[i]
  state<-racism$State.Abbreviation[i]
  racism$lat[i]<-cities1$Latitude[cities1$AccentCity==city&cities1$Region==state]
  racism$lon[i]<-cities1$Longitude[cities1$AccentCity==city&cities1$Region==state]
}

#create map
usa <- map_data("usa")

ggplot() + 
  geom_polygon(data = usa, aes(x=long, y = lat, group = group), fill = NA, color = "grey") + 
  coord_fixed(1.3)+
  geom_point(data=racism, aes(x=lon, y=lat, color=HC, size=Total.RHC...Numbers.across.6.years))+
  scale_colour_manual(values=c("black","red", "darkgreen", "orange"))+
  scale_size_continuous(name="Total number of Tweets",range = c(0.11, 6),breaks=5)+
  geom_text_repel(data=racism, aes(x=lon, y=lat,label=City,color=HC),size=6)+
  scale_colour_manual(values=c("red", "darkgreen", "orange"))+ 
  theme_bw()+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  theme(axis.title=element_blank(),axis.text=element_blank(),axis.ticks=element_blank(),panel.border = element_blank())+
  theme(legend.position="none")
