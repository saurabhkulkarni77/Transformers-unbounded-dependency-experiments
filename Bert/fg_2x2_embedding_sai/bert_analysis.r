rm(list=ls())

library(lme4)
library(lmerTest)
library(ggplot2)


####
#### Read in the dataset
####

options(scipen=999)

data <- read.delim("out_fg_emb_sai_combined_wordfinal_punct.txt",header=TRUE,sep =",")
summary(data)


####
#### Plot results
####

#data_1 <- data[which(data$EmbeddingLevel == 1),]

ggplot(aes(x = Condition, y = Surprisal, fill = Condition), data = data) +
 geom_boxplot() +
 xlab("") + #Conditions across four levels of clausal embedding") +
 facet_grid(.~EmbeddingLevel) +
   theme_bw(base_size=18)  +
   theme(legend.position = "none") +
  ggsave("~/Desktop/bert_sai.eps",height=6,width=16)

