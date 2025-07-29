library(psych)
M<-read.csv('/dockerfill_inputs.csv',header=TRUE,sep=",")
cohen.kappa(x=cbind(M$nature1,M$nature2))  #0.82
cohen.kappa(x=cbind(M$similarityld,M$similaritycs))  #0.82


M<-read.csv('all_results.csv',header=TRUE,sep=",")
cohen.kappa(x=cbind(M$naturalness1,M$naturalness2))  #0.82
cohen.kappa(x=cbind(M$similarity1,M$similarity2))  #0.82

M$model<-factor(M$model,c("CodeBERT","CodeSage","ContraCode","StarEncoder","UniXcoder","DockerFill"))

par(family = "Times")

temp_boxplot<- boxplot(M$avg_similarity~M$model, plot=FALSE)
default_width <- temp_boxplot$widths[1]
desired_fixed_width <- default_width / 2
relative_width <- desired_fixed_width / default_width

a<-boxplot(M$avg_similarity~M$model,xlab="",ylab = "Average similarity score",col=rgb(0.8, 0.4, 0.4, 0.5),width=rep(relative_width, 6))

boxplot(M$avg_naturalness~M$model)




ggplot(M, aes(y = avg_similarity, x = model)) +
  geom_boxplot(fill="white",width=0.45) +
  labs(x = "", y = "Averaged similarity score", title = "") +
theme_bw() +
  theme(
    axis.title.y = element_text(size = 13),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12)
  )



p<-ggplot(M, aes(y = avg_naturalness, x = model)) +
  geom_boxplot(fill="white",width=0.45) +
  labs(x = "", y = "Averaged naturalness score", title = "") +
  theme_bw() +
  theme(
    axis.title.y = element_text(size = 13),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12)
  )
p


summary(M$avg_similarity[M$model=="DockerFill"])
summary(M$avg_similarity[M$model=="CodeBERT"])
summary(M$avg_similarity[M$model=="CodeSage"])
summary(M$avg_similarity[M$model=="ContraCode"])
summary(M$avg_similarity[M$model=="StarEncoder"])
summary(M$avg_similarity[M$model=="UniXcoder"])


summary(M$avg_naturalness[M$model=="DockerFill"])
summary(M$avg_naturalness[M$model=="CodeBERT"])
summary(M$avg_naturalness[M$model=="CodeSage"])
summary(M$avg_naturalness[M$model=="ContraCode"])
summary(M$avg_naturalness[M$model=="StarEncoder"])
summary(M$avg_naturalness[M$model=="UniXcoder"])


library(effsize)
wilcox.test(M$avg_similarity[M$model=="DockerFill"],M$avg_similarity[M$model=="CodeBERT"])
cliff.delta(M$avg_similarity[M$model=="DockerFill"],M$avg_similarity[M$model=="CodeBERT"])

wilcox.test(M$avg_similarity[M$model=="DockerFill"],M$avg_similarity[M$model=="CodeSage"])
cliff.delta(M$avg_similarity[M$model=="DockerFill"],M$avg_similarity[M$model=="CodeSage"])

wilcox.test(M$avg_similarity[M$model=="DockerFill"],M$avg_similarity[M$model=="ContraCode"])
cliff.delta(M$avg_similarity[M$model=="DockerFill"],M$avg_similarity[M$model=="ContraCode"])

wilcox.test(M$avg_similarity[M$model=="DockerFill"],M$avg_similarity[M$model=="StarEncoder"])
cliff.delta(M$avg_similarity[M$model=="DockerFill"],M$avg_similarity[M$model=="StarEncoder"])

wilcox.test(M$avg_similarity[M$model=="DockerFill"],M$avg_similarity[M$model=="UniXcoder"])
cliff.delta(M$avg_similarity[M$model=="DockerFill"],M$avg_similarity[M$model=="UniXcoder"])





wilcox.test(M$avg_naturalness[M$model=="DockerFill"],M$avg_naturalness[M$model=="CodeBERT"])
cliff.delta(M$avg_naturalness[M$model=="DockerFill"],M$avg_naturalness[M$model=="CodeBERT"])

wilcox.test(M$avg_naturalness[M$model=="DockerFill"],M$avg_naturalness[M$model=="CodeSage"])
cliff.delta(M$avg_naturalness[M$model=="DockerFill"],M$avg_naturalness[M$model=="CodeSage"])

wilcox.test(M$avg_naturalness[M$model=="DockerFill"],M$avg_naturalness[M$model=="ContraCode"])
cliff.delta(M$avg_naturalness[M$model=="DockerFill"],M$avg_naturalness[M$model=="ContraCode"])

wilcox.test(M$avg_naturalness[M$model=="DockerFill"],M$avg_naturalness[M$model=="StarEncoder"])
cliff.delta(M$avg_naturalness[M$model=="DockerFill"],M$avg_naturalness[M$model=="StarEncoder"])

wilcox.test(M$avg_naturalness[M$model=="DockerFill"],M$avg_naturalness[M$model=="UniXcoder"])
cliff.delta(M$avg_naturalness[M$model=="DockerFill"],M$avg_naturalness[M$model=="UniXcoder"])


#install.packages("ggplot2")
#install.packages('rlang')
#install.packages('scales')
library(ggplot2)


group1 <- data.frame(
  CodeBERT = M$avg_similarity[M$model=="CodeBERT"],
  ContraCode = M$avg_similarity[M$model=="ContraCode"],
  CodeSage = M$avg_similarity[M$model=="CodeSage"],
  StarEncoder = M$avg_similarity[M$model=="StarEncoder"],
  UniXcoder = M$avg_similarity[M$model=="UniXcoder"],
  DockerFill = M$avg_similarity[M$model=="DockerFill"]
)

group2 <- data.frame(
  CodeBERT = M$avg_naturalness[M$model=="CodeBERT"],
  ContraCode = M$avg_naturalness[M$model=="ContraCode"],
  CodeSage = M$avg_naturalness[M$model=="CodeSage"],
  StarEncoder = M$avg_naturalness[M$model=="StarEncoder"],
  UniXcoder = M$avg_naturalness[M$model=="UniXcoder"],
  DockerFill = M$avg_naturalness[M$model=="DockerFill"]
)


#combined_data <- rbind(
#  data.frame(value = as.vector(t(group1)), group = "Similarity", Model = rep(colnames(group1), each = 372)),
#  data.frame(value = as.vector(t(group2)), group = "Naturalness", Model = rep(colnames(group2), each = 372))
#)

M<-read.csv('all_results_1.csv',header=TRUE,sep=",")

M$model<-factor(M$model,c("CodeBERT","ContraCode","CodeSage","StarEncoder","UniXcoder",
                          "DeepSeek-Coder","Code-Llama","Qwen-Coder","GPT-3.5-Turbo","GPT-4o","DockerFill"))

my_colors <- c("#E69F00", "#E69F00", "#E69F00", "#E69F00", "#E69F00", 
               "#56B4E9", "#56B4E9", "#56B4E9", "#56B4E9", "#56B4E9", 
               "#D55E56")

m<-ggplot(M, aes(x = model, y = avg_naturalness, fill = model
              )) +
  geom_boxplot(width=0.85) +
  labs(x = "", y = "Averaged naturalness score", title = "") +
  scale_fill_manual(values = my_colors)+
  theme_bw() +
  theme(legend.position = "none", 
        axis.title.y = element_text(size = 13),
        axis.text.x = element_text(size = 12,angle = 45, hjust = 1),
        axis.text.y = element_text(size = 12)) 

p<-ggplot(M, aes(x = model, y = avg_similarity, fill = model
)) +
  geom_boxplot(width=0.85) +
  labs(x = "", y = "Averaged similarity score", title = "") +
  scale_fill_manual(values = my_colors)+
  theme_bw() +
  theme(legend.position = "none", 
    #legend.direction = "horizontal",
    #legend.box = "horizontal", 
    axis.title.y = element_text(size = 13),
    #legend.text = element_text(size = 12),
    #legend.title = element_blank(),
    axis.text.x = element_text(size = 12,angle = 45, hjust = 1),
    axis.text.y = element_text(size = 12)) #+
p
