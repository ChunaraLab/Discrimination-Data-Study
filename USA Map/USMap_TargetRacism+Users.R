rm(list=ls())  
ls() 

# set directory
setwd( "/Users/kunalrelia/Desktop/Hate Crime data/Code/graphs/USMap")

#load libraries
library("ggplot2")
library(ggrepel)
library("maps")

#load racism and hate crime data and format city names
racism<-read.csv("racism_withUser.csv")
racism<-racism[1:100,2:12]
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

#Sort the data in ascending order based on proportion of racism Tweets 
racism<-racism[order(racism$Prevalence.of.Racism.in....x.10..2.),]

#Add racism Tweets categories
racism$RT<-""
racism$RT[1:25]<-"Low RT"
racism$RT[26:75]<-"Medium RT"
racism$RT[76:100]<-"High RT"

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

#racism$ratio=racism$RacismUsers/racism$RacismTweets
#racism$ratio[3]<-1
#racism$ratio[7]<-1

#create map
usa <- map_data("usa")

ggplot() + 
  geom_polygon(data = usa, aes(x=long, y = lat, group = group), fill = NA, color = "grey") + 
  coord_fixed(1.3)+
  geom_point(data=racism, aes(x=lon, y=lat, color=RT, size=ratio))+
  scale_colour_manual(values=c("black","red", "darkgreen", "orange"))+
  scale_size_continuous(name="Total number of Tweets",range = c(0.11, 6),breaks=5)+
  geom_text_repel(data=racism, aes(x=lon, y=lat,label=City,color=RT),size=6)+
  scale_colour_manual(values=c("red", "darkgreen", "orange"))+ 
  theme_bw()+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  theme(axis.title=element_blank(),axis.text=element_blank(),axis.ticks=element_blank(),panel.border = element_blank())+
  theme(legend.position="none")

#Underline the city names having proportion of racism that is targeted>0.5
for(i in 1:nrow(racism)){
  if(racism$Proportion.of.Racism.that.is.Targeted[i]>0.5){
    racism$City[i]<-sprintf("underline(%s)",gsub(" ", "~", racism$City[i], fixed = TRUE))
  } else{
    racism$City[i]<-sprintf("%s",gsub(" ", "~", racism$City[i], fixed = TRUE)) 
  }
}

#create the map with city names having proportion of racism that is targeted>0.5 underlined
ggplot() + 
  geom_polygon(data = usa, aes(x=long, y = lat, group = group), fill = NA, color = "grey") + 
  coord_fixed(1.3)+
  geom_point(data=racism, aes(x=lon, y=lat, color=RT, size=ratio))+
  #geom_point(data=racism[racism$Proportion.of.Racism.that.is.Targeted>0.5,], pch=21, fill=NA, stroke=3,aes(x=lon, y=lat, color=HC, size=ratio*1.5))+
  scale_colour_manual(values=c("black","red", "darkgreen", "orange"))+
  scale_size_continuous(name="Total number of Tweets",range = c(0.11, 6),breaks=5)+
  geom_text_repel(data=racism, aes(x=lon, y=lat,label=City,color=RT),size=6,parse=TRUE)+
  #geom_text_repel(data=racism[racism$Proportion.of.Racism.that.is.Targeted>0.5,], aes(x=lon, y=lat,label=parseLabel,color=HC),size=6,parse=TRUE)+
  #geom_text_repel(data=racism[racism$Proportion.of.Racism.that.is.Targeted<=0.5,], aes(x=lon, y=lat,label=City,color=HC),size=6)+
  scale_colour_manual(values=c("red", "darkgreen", "orange"))+ 
  theme_bw()+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  theme(axis.title=element_blank(),axis.text=element_blank(),axis.ticks=element_blank(),panel.border = element_blank())+
  theme(legend.position="none")
