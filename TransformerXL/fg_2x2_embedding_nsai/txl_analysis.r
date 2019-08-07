rm(list=ls())

library(lme4)
library(lmerTest)
library(ggplot2)


####
#### Read in the dataset
####

options(scipen=999)

data <- read.delim("out_fg_emb_txl_nsai.txt",header=TRUE,sep =",")
summary(data)


####
#### Plot results
####

ggplot(aes(x = Condition, y = Surprisal, fill = Condition), data = data) +
 geom_boxplot() +
 xlab("Conditions across four levels of clausal embedding") +
 facet_grid(.~EmbeddingLevel) +
    theme_bw(base_size=18)  +
 theme(legend.position = "none") + 
  ggsave("~/Desktop/txl_nsai.eps",height=6,width=16)
