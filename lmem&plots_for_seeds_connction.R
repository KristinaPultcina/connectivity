library(reshape2)
library(data.table)

library(lme4)
library(ggpubr)
library(emmeans)
library(lmerTest)

library(stringi)
library(gridExtra)
library(ggprism)

library(tibble)


library(dplyr)
library(purrr)
library(tidyverse)
options(scipen = 999)

path<- "/Volumes/My Passport for Mac/wpli_900_300_whole_brain"
row_names<- c('Anterior Cingulate and Medial Prefrontal Cortex-lh',
                  'Anterior Cingulate and Medial Prefrontal Cortex-rh',
                  'Auditory Association Cortex-lh',
                  'Auditory Association Cortex-rh',
                  'Dorsal Stream Visual Cortex-lh',
                  'Dorsal Stream Visual Cortex-rh',
                  'DorsoLateral Prefrontal Cortex-lh',
                  'DorsoLateral Prefrontal Cortex-rh',
                  'Early Auditory Cortex-lh',
                  'Early Auditory Cortex-rh',
                  'Early Visual Cortex-lh',
                  'Early Visual Cortex-rh',
                  'Inferior Frontal Cortex-lh',
                  'Inferior Frontal Cortex-rh',
                  'Inferior Parietal Cortex-lh',
                  'Inferior Parietal Cortex-rh',
                  'Insular and Frontal Opercular Cortex-lh',
                  'Insular and Frontal Opercular Cortex-rh',
                  'Lateral Temporal Cortex-lh',
                  'Lateral Temporal Cortex-rh',
                  'MT+ Complex and Neighboring Visual Areas-lh',
                  'MT+ Complex and Neighboring Visual Areas-rh',
                  'Medial Temporal Cortex-lh',
                  'Medial Temporal Cortex-rh',
                  'Orbital and Polar Frontal Cortex-lh',
                  'Orbital and Polar Frontal Cortex-rh',
                  'Paracentral Lobular and Mid Cingulate Cortex-lh',
                  'Paracentral Lobular and Mid Cingulate Cortex-rh',
                  'Posterior Cingulate Cortex-lh',
                  'Posterior Cingulate Cortex-rh',
                  'Posterior Opercular Cortex-lh',
                  'Posterior Opercular Cortex-rh',
                  'Premotor Cortex-lh',
                  'Premotor Cortex-rh',
                  'Primary Visual Cortex (V1)-lh',
                  'Primary Visual Cortex (V1)-rh',
                  'Somatosensory and Motor Cortex-lh',
                  'Somatosensory and Motor Cortex-rh',
                  'Superior Parietal Cortex-lh',
                  'Superior Parietal Cortex-rh',
                  'Temporo-Parieto-Occipital Junction-lh',
                  'Temporo-Parieto-Occipital Junction-rh',
                  'Ventral Stream Visual Cortex-lh',
                  'Ventral Stream Visual Cortex-rh')
col_names<-append(row_names,"full_filename")

full_df<- data.table()
df <-list.files(path,pattern = "*.txt",full.names =T)
for (d in df){
  print(d)
  f<- read.table(d,row.names=row_names)%>% mutate(filename = d)
  colnames(f)<-col_names
  filname<-rep(f$full_filename[1], 1936)
  long.format <-cor_gather(f[,1:44],drop.na = TRUE)
  long.format$full_filename<-filname
  full_df<-rbind(full_df,long.format)
}
  

             
df1 <- full_df %>% separate(full_filename, c("1","2","3","4","5","6", "short_filename"), sep="/")
##### don't forget to change the path ######
df1<-str_remove(full_df$full_filename, "/Volumes/My Passport for Mac/wpli_900_300_whole_brain/")
df1<-str_remove(df1, ".txt")
df1<-str_remove(df1, "_fb_cur")

df_new<- cbind(full_df, df1)
df_new$full_filename<- NULL
dF <- df_new %>% separate(df1, c("subject","round","trial_type","feedback"), sep="_")
#ips_lh<-dF$`Superior Parietal Cortex-lh`
#ips_rh<-dF$`Superior Parietal Cortex-rh`


autists_label_df <- filter(dF, subject %in% c('P301','P304','P307','P312','P313','P314','P316','P332',
                                                      'P321','P322','P323','P324','P325','P326','P327','P328',
                                                      'P329','P333','P334', 'P335','P338','P341','P342'))
autists_label_df $group<- "autists"



normal_label_df<- filter(dF, subject %in% c('P001','P004','P019','P021','P022','P032','P034','P035','P039',
                                                   'P040','P044','P047','P048','P053','P055','P058','P059','P060','P061',
                                                   'P063','P064','P065','P067'))
normal_label_df$group<- "normal"


df_for_lmem<-rbind(normal_label_df,autists_label_df)

df_for_lmem_acc_lh<-filter(df_for_lmem, var2=='Anterior Cingulate and Medial Prefrontal Cortex-rh' & var1=='Superior Parietal Cortex-lh')

m<- lmer(cor~ group*trial_type + (1|subject), data = df_for_lmem_acc_lh) # main part, fit model!
summary(m)
s <- step(m)
m2 <- get_model(s)
an <- NULL
an <- anova(m2)
an <- data.table(an,keep.rownames = TRUE)
an[`Pr(>F)`<0.001, stars:='***']
an[`Pr(>F)`<0.01 & `Pr(>F)`>0.001 , stars:='**']
an[`Pr(>F)`<0.05 & `Pr(>F)`>0.01 , stars:='*']
an[`Pr(>F)`>0.05 & `Pr(>F)`<0.1 , stars:='#']

sterr <- function(x) sd(x)/sqrt(length(x))
##### boxplot for all #######
emm_options(pbkrtest.limit = 40000)
marginal_em <- emmeans(m2, ~ as.factor(trial_type), level = 0.95)
marginal_em<- as.data.frame(marginal_em)

Tuk<- NULL
thr1 <- max(df_for_lmem_acc_lh[, mean(cor) + sterr(cor), by=c('trial_type')]$V1) 
thr1 <- thr1+0.02 #for RT

thr1_min <- min(df_for_lmem_acc_lh[!is.na(cor), mean(cor) - sterr(cor), by=c('trial_type')]$V1) 

Tuk<-data.table(summary(emmeans(m2, pairwise ~ trial_type, adjust = 'tukey',lmer.df = "satterthwaite"))$contrasts)
Tuk <- Tuk[, group1:=gsub(' -.*', '', contrast)][, group2:=gsub('.*- ', '', contrast)]
Tuk <- Tuk[p.value<0.1, p_significant:=format(p.value, digits = 3)]

n <- Tuk[!is.na(p_significant), .N]

Tuk[p.value<0.001, stars:='***']
Tuk[p.value<0.01 & p.value>0.001 , stars:='**']
Tuk[p.value<0.05 & p.value>0.01 , stars:='*']
Tuk[p.value>0.05 & p.value<0.1 , stars:='#']

plot_emmean<-ggplot(data = marginal_em, aes(x = factor(trial_type,level = c("norisk","prerisk","risk","postrisk")), 
                                             y = emmean,  ymin=emmean-SE, ymax = emmean+SE, group = 1))+
  scale_x_discrete(labels = c('HP','Pre-LP','LP', 'post-LP'))+
  geom_point() + geom_errorbar(width = 0.1, size=1.5)+geom_line(size=1.5)+labs(y = "wPLI", x = "Choice type")+
  theme_classic()+theme(text = element_text(size=20))+ ylim (0.20,1) +
  theme(axis.text.x = element_text(colour="black"), axis.text.y = element_text(colour="black"))+
  stat_pvalue_manual(Tuk, label = 'stars', size = 12, bracket.size = 1.5, tip.length = 0.01,y.position =c(0.95,0.99,0.89,0.97,0.9,0.84),inherit.aes = FALSE)

plot_emmean <- ggpar(plot_emmean,
                     ylim = c(0.20,1),
                     font.ytickslab = 30,
                     font.xtickslab = 27,
                     font.main = 25,
                     font.submain = 25,
                     font.x = 27,
                     font.y = 30)

plot_emmean

emm_options(pbkrtest.limit = 40000)
marginal_em <- emmeans(m2, ~ as.factor(trial_type|group), level = 0.95)
marginal_em<- as.data.frame(marginal_em)


Tuk<- NULL
thr1 <- max(df_for_lmem_acc_lh[, mean(cor) + sterr(cor), by=c('trial_type')]$V1) 
thr1 <- thr1+0.02 #for RT

thr1_min <- min(df_for_lmem_acc_lh[!is.na(cor), mean(cor) - sterr(cor), by=c('trial_type')]$V1) 

Tuk<-data.table(summary(emmeans(m2, pairwise ~ trial_type|group, adjust = 'tukey',lmer.df = "satterthwaite"))$contrasts)
Tuk <- Tuk[, group1:=gsub(' -.*', '', contrast)][, group2:=gsub('.*- ', '', contrast)]
Tuk <- Tuk[p.value<0.1, p_significant:=format(p.value, digits = 3)]

n <- Tuk[!is.na(p_significant), .N]

Tuk[p.value<0.001, stars:='***']
Tuk[p.value<0.01 & p.value>0.001 , stars:='**']
Tuk[p.value<0.05 & p.value>0.01 , stars:='*']
Tuk[p.value>0.05 & p.value<0.1 , stars:='#']

if (n>1){
  Tuk <- Tuk[!is.na(p_significant), y.position := seq((thr1+0.01), (thr1+0.3), 0.29/(n-1))]
} else {
  Tuk <- Tuk[!is.na(p_significant), y.position := thr1+0.1]
}
y.position<-Tuk$y.position

Tuk$emmean<-y.position

Tuk_aut<- filter(Tuk, group=="autists")
Tuk_norm<- filter(Tuk, group=="normal")
marginal_aut<-filter(marginal_em, group=="autists")
marginal_norm<-filter(marginal_em, group=="normal")

##### AUTISTS #######
plot_emmean<-ggplot(data = marginal_aut, aes(x = factor(trial_type,level = c("norisk","prerisk","risk","postrisk")), 
                                             y = emmean,  ymin=emmean-SE, ymax = emmean+SE, group = 1))+
  scale_x_discrete(labels = c('HP','Pre-LP','LP', 'post-LP'))+
  geom_point() + geom_errorbar(width = 0.1, size=1.5)+geom_line(size=1.5)+labs(y = "wPLI", x = "Choice type")+
  theme_classic()+theme(text = element_text(size=20))+ ylim (0.20,1) +
  theme(axis.text.x = element_text(colour="black"), axis.text.y = element_text(colour="black"))+
  geom_hline(yintercept= 0.0, linetype='dashed', col = 'black', size = 1.0)+
  stat_pvalue_manual(Tuk_aut, label = 'stars', size = 12, bracket.size = 1.5, tip.length = 0.01,y.position =c(0.95,0.99,0.89,0.97,0.9,0.85),inherit.aes = FALSE)

plot_emmean <- ggpar(plot_emmean,
                     ylim = c(0.20,1),
                     font.ytickslab = 30,
                     font.xtickslab = 27,
                     font.main = 25,
                     font.submain = 25,
                     font.x = 27,
                     font.y = 30)

plot_emmean
####### NORMAL ########3
plot_emmean<-ggplot(data = marginal_norm, aes(x = factor(trial_type,level = c("norisk","prerisk","risk","postrisk")), 
                                             y = emmean,  ymin=emmean-SE, ymax = emmean+SE, group = 1))+
  scale_x_discrete(labels = c('HP','Pre-LP','LP', 'post-LP'))+
  geom_point() + geom_errorbar(width = 0.1, size=1.5)+geom_line(size=1.5)+labs(y = "wPLI", x = "Choice type")+
  theme_classic()+theme(text = element_text(size=20))+ ylim (0.20,1) +
  theme(axis.text.x = element_text(colour="black"), axis.text.y = element_text(colour="black"))+
  geom_hline(yintercept= 0.0, linetype='dashed', col = 'black', size = 1.0)+
  stat_pvalue_manual(Tuk_norm, label = 'stars', size = 12, bracket.size = 1.5, tip.length = 0.01,y.position =c(0.95,0.99,0.89,0.97,0.9,0.84),inherit.aes = FALSE)

plot_emmean <- ggpar(plot_emmean,
                     ylim = c(0.20,1),
                     font.ytickslab = 30,
                     font.xtickslab = 27,
                     font.main = 25,
                     font.submain = 25,
                     font.x = 27,
                     font.y = 30)

plot_emmean


############# BETWEEN GROUP DIFFERENCES ##########
marginal_em <- emmeans(m2, ~ as.factor(trial_type|group), level = 0.95)
marginal_em<- as.data.frame(marginal_em)
Tuk<- NULL
thr1 <- max(df_for_lmem_acc_lh[, mean(cor) + sterr(cor), by=c('trial_type')]$V1) 
thr1 <- thr1+0.02 #for RT

thr1_min <- min(df_for_lmem_acc_lh[!is.na(cor), mean(cor) - sterr(cor), by=c('trial_type')]$V1) 

Tuk<-data.table(summary(emmeans(m2, pairwise ~ group|trial_type, adjust = 'tukey',lmer.df = "satterthwaite"))$contrasts)
Tuk <- Tuk[, group1:=gsub(' -.*', '', contrast)][, group2:=gsub('.*- ', '', contrast)]
Tuk <- Tuk[p.value<0.1, p_significant:=format(p.value, digits = 3)]

n <- Tuk[!is.na(p_significant), .N]

Tuk[p.value<0.001, stars:='***']
Tuk[p.value<0.01 & p.value>0.001 , stars:='**']
Tuk[p.value<0.05 & p.value>0.01 , stars:='*']
Tuk[p.value>0.05 & p.value<0.1 , stars:='#']

signif <- Tuk[!is.na(stars)]

sequence <-data.table(trial_type=c("norisk","prerisk","risk", "postrisk"),number=c(1,2,3,4))


y_values_rew <- df_for_lmem_acc_lh[group == 'normal',
                   mean(cor)+sterr(cor)+0.05, by='trial_type']
setnames(y_values_rew,'V1','y_values_rew')

y_values_lose <-  df_for_lmem_acc_lh[group == 'autists',
                     mean(cor)+sterr(cor)+0.05, by='trial_type']

setnames(y_values_lose,'V1','y_values_lose')

y_values <- merge(y_values_lose,y_values_rew,by='trial_type')
y_values <- merge(y_values,sequence,by='trial_type')
y_values[,y_max:=max(y_values_lose,y_values_rew),by=trial_type]
y_values[,y_min:=min(y_values_lose,y_values_rew),by=trial_type]


y_values <- merge(y_values,signif,by='trial_type')
#y_values<- filter(y_values, feedback_cur=="negative")

setnames(marginal_em, 'y_values_lose', "emmean")

p1 <- ggplot(marginal_em, aes(x = factor(trial_type,level = c("norisk","prerisk","risk","postrisk")),
                              y = emmean,  ymin=emmean-SE, ymax = emmean+SE, color = group,group = group))+
  scale_x_discrete(labels = c('HP','Pre-LP','LP', 'post-LP'))+ geom_line(size=1.5)+
  geom_point(position=position_dodge(0.1)) + geom_errorbar(width = 0.1,  position=position_dodge(0.1), size=1.5)+labs(y = "wPLI", x = "Choice type")+
  theme_classic()+ theme(text = element_text(size=20))+scale_color_discrete(name = "Group", labels = c("Autists", "Normotypical"))+theme(legend.position="bottom")+
  ylim(0.20,1) +
  geom_hline(yintercept=-0.0, linetype='dashed', col = 'black', size = 1.0)+
  theme(axis.text.x = element_text(colour="black"), axis.text.y = element_text(colour="black"))+
  scale_color_manual(values=c("magenta","gold1"))

p1 <- p1+geom_signif(y_position=c(y_values$y_max +0.05),
                     xmin=c(y_values$number-0.075), xmax=c(y_values$number+0.075),
                     annotation=c(y_values$stars),col='black',
                     tip_length=0.003,textsize = 12,vjust = 0.4,size = 1.2)
p1<- ggpar(p1,
           ylim = c(0.4,1),
           font.ytickslab = 30,
           font.xtickslab = 27,
           font.main = 25,
           font.submain = 25,
           font.x = 27,
           font.y = 27)


p1



marginal_em <- emmeans(m2, ~ as.factor(trial_type|feedback|group), level = 0.95)
marginal_em<- as.data.frame(marginal_em)
marginal_positive<-filter(marginal_em, feedback=="positive")
marginal_negative<-filter(marginal_em, feedback=="negative")
Tuk<- NULL
thr1 <- max(df_for_lmem_acc_lh[, mean(cor) + sterr(cor), by=c('trial_type')]$V1) 
thr1 <- thr1+0.02 #for RT

thr1_min <- min(df_for_lmem_acc_lh[!is.na(cor), mean(cor) - sterr(cor), by=c('trial_type')]$V1) 

Tuk<-data.table(summary(emmeans(m2, pairwise ~ group|feedback|trial_type, adjust = 'tukey',lmer.df = "satterthwaite"))$contrasts)
Tuk <- Tuk[, group1:=gsub(' -.*', '', contrast)][, group2:=gsub('.*- ', '', contrast)]
Tuk <- Tuk[p.value<0.1, p_significant:=format(p.value, digits = 3)]

n <- Tuk[!is.na(p_significant), .N]

Tuk[p.value<0.001, stars:='***']
Tuk[p.value<0.01 & p.value>0.001 , stars:='**']
Tuk[p.value<0.05 & p.value>0.01 , stars:='*']
Tuk[p.value>0.05 & p.value<0.1 , stars:='#']



Tuk_positive<- filter(Tuk, feedback=="positive")
Tuk_negative<-  filter(Tuk, feedback=="negative")
signif_pos <- Tuk_positive[!is.na(stars)]
signif_neg <- Tuk_negative[!is.na(stars)]
sequence <-data.table(trial_type=c("norisk","prerisk","risk", "postrisk"),number=c(1,2,3,4))

####### between group differences positive feedback #####
y_values_rew <- df_for_lmem_acc_lh[feedback == 'positive',
                   mean(cor)+sterr(cor)+0.15, by='trial_type']
setnames(y_values_rew,'V1','y_values_rew')

y_values_lose <-  df_for_lmem_acc_lh[feedback == 'negative',
                     mean(cor)+sterr(cor)+0.05, by='trial_type']

setnames(y_values_lose,'V1','y_values_lose')

y_values <- merge(y_values_lose,y_values_rew,by='trial_type')
y_values <- merge(y_values,sequence,by='trial_type')
y_values[,y_max:=max(y_values_lose,y_values_rew),by=trial_type]
y_values[,y_min:=min(y_values_lose,y_values_rew),by=trial_type]

y_values <- merge(y_values,signif_pos,by='trial_type')

setnames(marginal_positive, 'y_values_rew', "emmean")

p1 <- ggplot(marginal_positive, aes(x = factor(trial_type,level = c("norisk","prerisk","risk","postrisk")),
                                    y = emmean,  ymin=emmean-SE, ymax = emmean+SE, color = group,group = group))+
  scale_x_discrete(labels = c('HP','Pre-LP','LP', 'post-LP'))+ geom_line(size=1.5)+
  geom_point(position=position_dodge(0.1)) + geom_errorbar(width = 0.1,  position=position_dodge(0.1), size=1.5)+labs(y = "wPLI", x = "Choice type")+
  theme_classic()+ theme(text = element_text(size=20))+scale_color_discrete(name = "Group", labels = c("Autists", "Normotypical"))+theme(legend.position="bottom")+
  ylim(0.20,1) +
  theme(axis.text.x = element_text(colour="black"), axis.text.y = element_text(colour="black"))+
  scale_color_manual(values=c("magenta","gold1"))
p1 <- p1+geom_signif(y_position=c(y_values$y_max +0.05),
                     xmin=c(y_values$number-0.025), xmax=c(y_values$number+0.075),
                     annotation=c(y_values$stars),col='black',
                     tip_length=0.003,textsize = 12,vjust = 0.4,size = 1.2)

p1<- ggpar(p1,
           ylim = c(0.20,1),
           font.ytickslab = 30,
           font.xtickslab = 27,
           font.main = 25,
           font.submain = 25,
           font.x = 27,
           font.y = 27)

p1

####### between group differences negative feedback #####
y_values_rew <- df_for_lmem_acc_lh[feedback == 'positive',
                                   mean(cor)+sterr(cor)+0.1, by='trial_type']
setnames(y_values_rew,'V1','y_values_rew')

y_values_lose <-  df_for_lmem_acc_lh[feedback == 'negative',
                                     mean(cor)+sterr(cor)+0.1, by='trial_type']

setnames(y_values_lose,'V1','y_values_lose')

y_values <- merge(y_values_lose,y_values_rew,by='trial_type')
y_values <- merge(y_values,sequence,by='trial_type')
y_values[,y_max:=max(y_values_lose,y_values_rew),by=trial_type]
y_values[,y_min:=min(y_values_lose,y_values_rew),by=trial_type]

# ylim1 <- min(y_values$y_min)
# ylim2 <- max(y_values$y_max)

y_values <- merge(y_values,signif_neg,by='trial_type')

setnames(marginal_negative, 'y_values_lose', "emmean")


p1 <- ggplot(marginal_negative, aes(x = factor(trial_type,level = c("norisk","prerisk","risk","postrisk")),
                                    y = emmean,  ymin=emmean-SE, ymax = emmean+SE, color = group,group = group))+
  scale_x_discrete(labels = c('HP','Pre-LP','LP', 'post-LP'))+ geom_line(size=1.5)+
  geom_point(position=position_dodge(0.1)) + geom_errorbar(width = 0.1,  position=position_dodge(0.1), size=1.5)+labs(y = "wPLI", x = "Choice type")+
  theme_classic()+ theme(text = element_text(size=20))+scale_color_discrete(name = "Group", labels = c("Autists", "Normotypical"))+theme(legend.position="bottom")+
  ylim(0.20,1) +
  geom_hline(yintercept=-0.0, linetype='dashed', col = 'black', size = 1.0)+
  theme(axis.text.x = element_text(colour="black"), axis.text.y = element_text(colour="black"))+
  scale_color_manual(values=c("magenta","gold1"))
p1 <- p1+geom_signif(y_position=c(y_values$y_max +0.05),
                     xmin=c(y_values$number-0.075), xmax=c(y_values$number+0.075),
                     annotation=c(y_values$stars),col='black',
                     tip_length=0.003,textsize = 12,vjust = 0.4,size = 1.2)

p1<- ggpar(p1,
           ylim = c(0.20,1),
           font.ytickslab = 30,
           font.xtickslab = 27,
           font.main = 25,
           font.submain = 25,
           font.x = 27,
           font.y = 27)

p1


########### InGROUP FEEDBACK DIFFERENCES ###########

emm_options(pbkrtest.limit = 40000)
marginal_em <- emmeans(m2, ~ as.factor(trial_type|group|feedback), level = 0.99)
marginal_em<- as.data.frame(marginal_em)

marginal_norm<-filter(marginal_em, group=="normal")
marginal_autists<-filter(marginal_em, group=="autists")

Tuk<- NULL
thr1 <- max(df_for_lmem_acc_lh[, mean(cor) + sterr(cor), by=c('trial_type')]$V1) 
thr1 <- thr1+0.02 #for RT

thr1_min <- min(df_for_lmem_acc_lh[!is.na(cor), mean(cor) - sterr(cor), by=c('trial_type')]$V1) 

Tuk<-data.table(summary(emmeans(m2, pairwise ~ feedback|group|trial_type, adjust = 'tukey',lmer.df = "satterthwaite"))$contrasts)
Tuk <- Tuk[, group1:=gsub(' -.*', '', contrast)][, group2:=gsub('.*- ', '', contrast)]
Tuk <- Tuk[p.value<0.1, p_significant:=format(p.value, digits = 3)]

n <- Tuk[!is.na(p_significant), .N]

Tuk[p.value<0.001, stars:='***']
Tuk[p.value<0.01 & p.value>0.001 , stars:='**']
Tuk[p.value<0.05 & p.value>0.01 , stars:='*']
#Tuk[p.value>0.05 & p.value<0.1 , stars:='#']


Tuk<- filter(Tuk, group=="normal")
signif <- Tuk[!is.na(stars)]

sequence <-data.table(trial_type=c("norisk","prerisk","risk", "postrisk"),number=c(1,2,3,4))
sterr <- function(x) sd(x)/sqrt(length(x))

y_values_rew <- df_for_lmem_acc_lh[feedback == 'positive',
                   mean(cor)+sterr(cor)+0.1, by='trial_type']
setnames(y_values_rew,'V1','y_values_rew')

y_values_lose <-  df_for_lmem_acc_lh[feedback == 'negative',
                     mean(cor)+sterr(cor)+0.1, by='trial_type']

setnames(y_values_lose,'V1','y_values_lose')

y_values <- merge(y_values_lose,y_values_rew,by='trial_type')
y_values <- merge(y_values,sequence,by='trial_type')
y_values[,y_max:=max(y_values_lose,y_values_rew),by=trial_type]
y_values[,y_min:=min(y_values_lose,y_values_rew),by=trial_type]

# ylim1 <- min(y_values$y_min)
# ylim2 <- max(y_values$y_max)

y_values <- merge(y_values,signif,by='trial_type')

setnames(marginal_norm, 'y_values_lose', "emmean")

p1 <- ggplot(marginal_norm, aes(x = factor(trial_type,level = c("norisk","prerisk","risk","postrisk")),
                                y = emmean,  ymin=emmean-SE, ymax = emmean+SE, color = feedback,group = feedback))+
  scale_x_discrete(labels = c('HP','Pre-LP','LP', 'post-LP'))+ geom_line(size=1.5)+
  geom_point(position=position_dodge(0.1)) + geom_errorbar(width = 0.1,  position=position_dodge(0.1), size=1.5)+labs(y = "wPLI", x = "Choice type")+
  theme_classic()+ theme(text = element_text(size=20))+scale_color_discrete(name = "Current feedback", labels = c("Loss", "Gain"))+theme(legend.position="bottom")+
  ylim(0.20,1.2) +
  geom_hline(yintercept=-0.0, linetype='dashed', col = 'black', size = 1.0)+
  theme(axis.text.x = element_text(colour="black"), axis.text.y = element_text(colour="black"))

p1 <- p1+geom_signif(y_position=c(y_values$y_max + 0.05),
                     xmin=c(y_values$number-0.075), xmax=c(y_values$number+0.075),
                     annotation=c(y_values$stars),col='black',
                     tip_length=0.003,textsize = 12,vjust = 0.4,size = 1.2)

p1<- ggpar(p1,
           ylim = c(0.20,1.2),
           font.ytickslab = 30,
           font.xtickslab = 27,
           font.main = 25,
           font.submain = 25,
           font.x = 27,
           font.y = 27)

p1


Tuk<- NULL
thr1 <- max(df_for_lmem_acc_lh[, mean(cor) + sterr(cor), by=c('trial_type')]$V1) 
thr1 <- thr1+0.02 #for RT

thr1_min <- min(df_for_lmem_acc_lh[!is.na(cor), mean(cor) - sterr(cor), by=c('trial_type')]$V1) 

Tuk<-data.table(summary(emmeans(m2, pairwise ~ feedback|group|trial_type, adjust = 'tukey',lmer.df = "satterthwaite"))$contrasts)
Tuk <- Tuk[, group1:=gsub(' -.*', '', contrast)][, group2:=gsub('.*- ', '', contrast)]
Tuk <- Tuk[p.value<0.1, p_significant:=format(p.value, digits = 3)]

n <- Tuk[!is.na(p_significant), .N]

Tuk[p.value<0.001, stars:='***']
Tuk[p.value<0.01 & p.value>0.001 , stars:='**']
Tuk[p.value<0.05 & p.value>0.01 , stars:='*']
Tuk[p.value>0.05 & p.value<0.1 , stars:='#']


Tuk<- filter(Tuk, group=="autists")
signif <- Tuk[!is.na(stars)]

sequence <-data.table(trial_type=c("norisk","prerisk","risk", "postrisk"),number=c(1,2,3,4))
sterr <- function(x) sd(x)/sqrt(length(x))

y_values_rew <- df_for_lmem_acc_lh[feedback == 'positive',
                                   mean(cor)+sterr(cor)+0.1, by='trial_type']
setnames(y_values_rew,'V1','y_values_rew')

y_values_lose <-  df_for_lmem_acc_lh[feedback == 'negative',
                                     mean(cor)+sterr(cor)+0.1, by='trial_type']

setnames(y_values_lose,'V1','y_values_lose')

y_values <- merge(y_values_lose,y_values_rew,by='trial_type')
y_values <- merge(y_values,sequence,by='trial_type')
y_values[,y_max:=max(y_values_lose,y_values_rew),by=trial_type]
y_values[,y_min:=min(y_values_lose,y_values_rew),by=trial_type]


y_values <- merge(y_values,signif,by='trial_type')

setnames(marginal_autists, 'y_values_lose', "emmean")

p1 <- ggplot(marginal_autists, aes(x = factor(trial_type,level = c("norisk","prerisk","risk","postrisk")),
                                y = emmean,  ymin=emmean-SE, ymax = emmean+SE, color = feedback,group = feedback))+
  scale_x_discrete(labels = c('HP','Pre-LP','LP', 'post-LP'))+ geom_line(size=1.5)+
  geom_point(position=position_dodge(0.1)) + geom_errorbar(width = 0.1,  position=position_dodge(0.1), size=1.5)+labs(y = "wPLI", x = "Choice type")+
  theme_classic()+ theme(text = element_text(size=20))+scale_color_discrete(name = "Current feedback", labels = c("Loss", "Gain"))+theme(legend.position="bottom")+
  ylim(0.20,1.2) +
  geom_hline(yintercept=-0.0, linetype='dashed', col = 'black', size = 1.0)+
  theme(axis.text.x = element_text(colour="black"), axis.text.y = element_text(colour="black"))

p1 <- p1+geom_signif(y_position=c(y_values$y_max + 0.05),
                     xmin=c(y_values$number-0.075), xmax=c(y_values$number+0.075),
                     annotation=c(y_values$stars),col='black',
                     tip_length=0.003,textsize = 12,vjust = 0.4,size = 1.2)

p1<- ggpar(p1,
           ylim = c(0.20,1.2),
           font.ytickslab = 30,
           font.xtickslab = 27,
           font.main = 25,
           font.submain = 25,
           font.x = 27,
           font.y = 27)

p1



marginal_em <- emmeans(m2, ~ as.factor(trial_type|feedback), level = 0.95)
marginal_em<- as.data.frame(marginal_em)
Tuk<- NULL
thr1 <- max(df_for_lmem_acc_lh[, mean(cor) + sterr(cor), by=c('trial_type')]$V1) 
thr1 <- thr1+0.02 #for RT

thr1_min <- min(df_for_lmem_acc_lh[!is.na(cor), mean(cor) - sterr(cor), by=c('trial_type')]$V1) 

Tuk<-data.table(summary(emmeans(m2, pairwise ~ feedback|trial_type, adjust = 'tukey',lmer.df = "satterthwaite"))$contrasts)
Tuk <- Tuk[, group1:=gsub(' -.*', '', contrast)][, group2:=gsub('.*- ', '', contrast)]
Tuk <- Tuk[p.value<0.1, p_significant:=format(p.value, digits = 3)]

n <- Tuk[!is.na(p_significant), .N]

Tuk[p.value<0.001, stars:='***']
Tuk[p.value<0.01 & p.value>0.001 , stars:='**']
Tuk[p.value<0.05 & p.value>0.01 , stars:='*']
#Tuk[p.value>0.05 & p.value<0.1 , stars:='#']


signif <- Tuk[!is.na(stars)]

sequence <-data.table(trial_type=c("norisk","prerisk","risk", "postrisk"),number=c(1,2,3,4))
sterr <- function(x) sd(x)/sqrt(length(x))

y_values_rew <- df_for_lmem_acc_lh[feedback == 'positive',
                                   mean(cor)+sterr(cor)+0.1, by='trial_type']
setnames(y_values_rew,'V1','y_values_rew')

y_values_lose <-  df_for_lmem_acc_lh[feedback == 'negative',
                                     mean(cor)+sterr(cor)+0.1, by='trial_type']

setnames(y_values_lose,'V1','y_values_lose')

y_values <- merge(y_values_lose,y_values_rew,by='trial_type')
y_values <- merge(y_values,sequence,by='trial_type')
y_values[,y_max:=max(y_values_lose,y_values_rew),by=trial_type]
y_values[,y_min:=min(y_values_lose,y_values_rew),by=trial_type]

# ylim1 <- min(y_values$y_min)
# ylim2 <- max(y_values$y_max)

y_values <- merge(y_values,signif,by='trial_type')

setnames(marginal_em, 'y_values_lose', "emmean")

p1 <- ggplot(marginal_em, aes(x = factor(trial_type,level = c("norisk","prerisk","risk","postrisk")),
                                   y = emmean,  ymin=emmean-SE, ymax = emmean+SE, color = feedback,group = feedback))+
  scale_x_discrete(labels = c('HP','Pre-LP','LP', 'post-LP'))+ geom_line(size=1.5)+
  geom_point(position=position_dodge(0.1)) + geom_errorbar(width = 0.1,  position=position_dodge(0.1), size=1.5)+labs(y = "wPLI", x = "Choice type")+
  theme_classic()+ theme(text = element_text(size=20))+scale_color_discrete(name = "Current feedback", labels = c("Loss", "Gain"))+theme(legend.position="bottom")+
  ylim(0.20,1.2) +
  geom_hline(yintercept=-0.0, linetype='dashed', col = 'black', size = 1.0)+
  theme(axis.text.x = element_text(colour="black"), axis.text.y = element_text(colour="black"))

p1 <- p1+geom_signif(y_position=c(y_values$y_max + 0.05),
                     xmin=c(y_values$number-0.075), xmax=c(y_values$number+0.075),
                     annotation=c(y_values$stars),col='black',
                     tip_length=0.003,textsize = 12,vjust = 0.4,size = 1.2)

p1<- ggpar(p1,
           ylim = c(0.20,1.2),
           font.ytickslab = 30,
           font.xtickslab = 27,
           font.main = 25,
           font.submain = 25,
           font.x = 27,
           font.y = 27)

p1

