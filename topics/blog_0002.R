#title: dynamic variable name in lm model and ggplot2
#author: Jiqi Liu
#date: 2020-08-02

#library packages
library(ggplot2)
library(tidyverse)

#make a dataset

x <- 1:10
Class <- c("A","B","C","D")

df <- data.frame()
count = 1
for (i in Class) {
  for (j in 1:6) {
    y1 <- sample(1:10, 1) * x + rnorm(10,0,1)
    y2 <- sample(1:10, 1) * x + rnorm(10,0,1)
    y3 <- sample(1:10, 1) * x + rnorm(10,0,1)
    y4 <- sample(1:10, 1) * x + rnorm(10,0,1)
    temp <- data.frame(x = x, y1 = y1, y2 = y2, y3 = y3, y4 = y4)
    temp$Class <- i
    temp$ID <- j * count
    df <- rbind(df, temp)
  }
  count = count + 1
}
df$ID <- as.character(df$ID)
str(df)
head(df)
unique(df$Class)
unique(df$ID)
#save the dataset example
#write.csv(df,"data/example_blog0002.csv",quote = F, row.names = F)

#visulazation in ggplot

names(df)

i <- 2

df %>%
  ggplot() +
  geom_smooth(aes_string(x = 'x', y = names(df)[i]), color = 'red', size = 2) +
  geom_point(aes_string(x = "x", y = names(df)[i], color = "ID")) +
  facet_wrap(~Class, nrow = 2, scales = 'free_y') +
  theme(
    strip.text = element_text(size = 15),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    title = element_text(size = 18),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    plot.title = element_text(hjust = 0.5),
    panel.background = element_rect(fill = "white",colour = "white",size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'dashed',colour = "grey"), 
    panel.grid.minor = element_line(size = 0.25, linetype = 'dashed',colour = "grey"),
    panel.border = element_rect(colour = "black", fill = NA, size = 0.5),
    legend.position = 'top')

## lm model

df %>%
  group_by(ID) %>%
  summarise(slope = lm(formula(paste0(names(df)[i],"~ x")))$coefficients[2]) %>%
  left_join(select(df, Class, ID)) %>%
  ggplot() +
  geom_boxplot(aes(x = Class, y = slope)) +
  theme(
    strip.text = element_text(size = 15),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    title = element_text(size = 18),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    plot.title = element_text(hjust = 0.5),
    panel.background = element_rect(fill = "white",colour = "white",size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'dashed',colour = "grey"), 
    panel.grid.minor = element_line(size = 0.25, linetype = 'dashed',colour = "grey"),
    panel.border = element_rect(colour = "black", fill = NA, size = 0.5))
  
