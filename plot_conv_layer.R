d <- dir("/Users/johan.renaudie/Documents/Manuscripts/Programming/!Ryan MS/ryan_project/environment/visualization2",pattern="csv$")
D <- dir("/Users/johan.renaudie/Documents/Manuscripts/Programming/!Ryan MS/ryan_project/environment/visualization2",pattern="csv$",full.names=TRUE)
L102 <- lapply(D,read.table,sep="\t")
d <- gsub("_278","278",d)
S <- do.call(rbind,strsplit(d,"_"))
pcl <- data.frame(Pic=S[,1],Conv=as.integer(unlist(regmatches(S[,2],gregexpr("[0-9]+",S[,2])))),Layer=as.integer(unlist(regmatches(S[,3],gregexpr("[0-9]+",S[,3])))),stringsAsFactors=FALSE)
n_sp <- unique(pcl[,1])
n_cv <- sort(unique(pcl[,2]))
n_ly <- sort(unique(pcl[,3]))
res <- list()
for(i in seq_along(n_sp)){
  res[[i]]<-list()
  for(j in seq_along(n_cv)){
    sub <- L102[pcl[,1]==n_sp[i]&pcl[,2]==n_cv[j]]
    res[[i]][[j]]<-array(dim=c(nrow(sub[[1]]),ncol(sub[[1]]),length(n_ly)))
    for(k in seq_along(n_ly)){
      res[[i]][[j]][,,k] <- as.matrix(L102[pcl[,1]==n_sp[i]&pcl[,2]==n_cv[j]&pcl[,3]==n_ly[k]][[1]])
    }
  }
}
dir.create("plots")
setwd("plots")
for(i in seq_along(n_sp)){
  for(j in seq_along(n_cv)){
    setEPS(width=4,height=8,paper="special")
    postscript(sprintf("%s_conv%i.eps",n_sp[i],n_cv[j]))
    par(mfrow=c(8,4))
    for(k in 1:32){
      par(mar=c(0,0,0,0))
      image(res[[i]][[j]][,,k],ax=F,an=F,col=heat.colors(13))
      box(lwd=2)
    }
    dev.off()
  }
}
