library(data.table)
library(stringr)

data = fread("train.csv",encoding = "UTF-8")
strsplit_amenities = function(x){
  splitted = strsplit(x,"\"\"|\\{|\\}|,")[[1]] # remove useless brackets or etc..
  splitted = splitted[splitted!=""&!str_detect(splitted,"translation missing")] #remove blanks or translation missing
  if(length(splitted)>0) return(splitted)
  else return(NA)
}
splitted = lapply(data$amenities,strsplit_amenities)
splitted_vector = unlist(splitted)
unique(splitted_vector)
amenities_df = rbindlist(lapply(splitted, as.data.frame.list),fill=TRUE) # amenities to data frame
colnames(amenities_df) = unlist(lapply(colnames(amenities_df),function(x){return (substring(x,first=3))})) # rename columns
amenities_df = apply(amenities_df,MARGIN = c(1,2),function(x){return(ifelse(is.na(x),FALSE,TRUE))}) #convert to TRUE/FALSE data frame
colnames(amenities_df)[which(colnames(amenities_df)=="24.hour.check.in.")] = "always.check.in" #can make some bugs
head(amenities_df[,seq(1,10)])
write.csv(amenities_df,"amenities.csv")
