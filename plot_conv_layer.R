d <- dir("environment/visualization/layers",pattern="csv$")
D <- dir("environment/visualization/layers",pattern="csv$",full.names=TRUE)
L102 <- lapply(D,read.table,sep="\t")
d <- gsub("_278","278",d)
S <- do.call(rbind,strsplit(d,"_"))
pcl <- data.frame(Pic=S[,1],
                  Conv=as.integer(unlist(regmatches(S[,2],gregexpr("[0-9]+",S[,2])))),
                  Layer=as.integer(unlist(regmatches(S[,3],gregexpr("[0-9]+",S[,3])))),
                  stringsAsFactors=FALSE)
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
pal <- colorRampPalette(c("white","#fdae6b","#e6550d"))
n <- 100
if(!dir.exists("environment/visualization/")) dir.create("environment/visualization/plots")
for(i in seq_along(n_sp)){
  for(j in seq_along(n_cv)){
    setEPS(width=4,height=8,paper="special")
    postscript(sprintf("environment/visualization/plots/%s_conv%i.eps",n_sp[i],n_cv[j]))
    par(mfrow=c(8,4))
    for(k in 1:32){
      par(mar=c(0,0,0,0))
      image(res[[i]][[j]][,,k],ax=F,an=F,col=pal(n),breaks=seq(0,6,len=n+1))
      box(lwd=2)
    }
    dev.off()
  }
}
dev.new(width=7,height=2)
par(xaxs="i",yaxs="i");frame();rect(0:(n-1)/n,0,1:n/n,1,col=pal(n),border=NA);axis(1,at=seq(0,1,len=7),lab=0:6,lwd.ticks=1,lwd=0);mtext("ReLu6 units",1,2);box()
dev.copy(pdf,"scale.pdf",width=7,height=2)
