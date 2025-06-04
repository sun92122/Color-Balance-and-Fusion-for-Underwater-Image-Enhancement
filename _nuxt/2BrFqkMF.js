import{I as d,J as f,a7 as m,aU as h,g as v,X as o,x as r,B as a,Y as b,a0 as i,Z as L,$ as y,a3 as S,a4 as w}from"./vUeD5rKf.js";import{Z as s}from"./DGEOgZpe.js";import{s as g}from"./C6CB1oH_.js";import k from"./RivuQeVp.js";import"./CkHEsmfq.js";import"./Be1fzYNM.js";import"./B-QYTNNy.js";import"./F3IPWLaN.js";import"./DFoE_xXZ.js";import"./DQ3OPcba.js";import"./DmfvTtO6.js";var E=d`
    .p-scrolltop.p-button {
        position: fixed !important;
        inset-block-end: 20px;
        inset-inline-end: 20px;
    }

    .p-scrolltop-sticky.p-button {
        position: sticky !important;
        display: flex;
        margin-inline-start: auto;
    }

    .p-scrolltop-enter-from {
        opacity: 0;
    }

    .p-scrolltop-enter-active {
        transition: opacity 0.15s;
    }

    .p-scrolltop.p-scrolltop-leave-to {
        opacity: 0;
    }

    .p-scrolltop-leave-active {
        transition: opacity 0.15s;
    }
`,$={root:function(t){var l=t.props;return["p-scrolltop",{"p-scrolltop-sticky":l.target!=="window"}]},icon:"p-scrolltop-icon"},C=f.extend({name:"scrolltop",style:E,classes:$}),T={name:"BaseScrollTop",extends:m,props:{target:{type:String,default:"window"},threshold:{type:Number,default:400},icon:{type:String,default:void 0},behavior:{type:String,default:"smooth"},buttonProps:{type:Object,default:function(){return{rounded:!0}}}},style:C,provide:function(){return{$pcScrollTop:this,$parentInstance:this}}},A={name:"ScrollTop",extends:T,inheritAttrs:!1,scrollListener:null,container:null,data:function(){return{visible:!1}},mounted:function(){this.target==="window"?this.bindDocumentScrollListener():this.target==="parent"&&this.bindParentScrollListener()},beforeUnmount:function(){this.target==="window"?this.unbindDocumentScrollListener():this.target==="parent"&&this.unbindParentScrollListener(),this.container&&(s.clear(this.container),this.overlay=null)},methods:{onClick:function(){var t=this.target==="window"?window:this.$el.parentElement;t.scroll({top:0,behavior:this.behavior})},checkVisibility:function(t){t>this.threshold?this.visible=!0:this.visible=!1},bindParentScrollListener:function(){var t=this;this.scrollListener=function(){t.checkVisibility(t.$el.parentElement.scrollTop)},this.$el.parentElement.addEventListener("scroll",this.scrollListener)},bindDocumentScrollListener:function(){var t=this;this.scrollListener=function(){t.checkVisibility(h())},window.addEventListener("scroll",this.scrollListener)},unbindParentScrollListener:function(){this.scrollListener&&(this.$el.parentElement.removeEventListener("scroll",this.scrollListener),this.scrollListener=null)},unbindDocumentScrollListener:function(){this.scrollListener&&(window.removeEventListener("scroll",this.scrollListener),this.scrollListener=null)},onEnter:function(t){s.set("overlay",t,this.$primevue.config.zIndex.overlay)},onAfterLeave:function(t){s.clear(t)},containerRef:function(t){this.container=t?t.$el:void 0}},computed:{scrollTopAriaLabel:function(){return this.$primevue.config.locale.aria?this.$primevue.config.locale.aria.scrollTop:void 0}},components:{ChevronUpIcon:g,Button:k}};function P(e,t,l,B,c,n){var p=v("Button");return r(),o(w,i({name:"p-scrolltop",appear:"",onEnter:n.onEnter,onAfterLeave:n.onAfterLeave},e.ptm("transition")),{default:a(function(){return[c.visible?(r(),o(p,i({key:0,ref:n.containerRef,class:e.cx("root"),onClick:n.onClick,"aria-label":n.scrollTopAriaLabel,unstyled:e.unstyled},e.buttonProps,{pt:e.ptm("root")}),{icon:a(function(u){return[L(e.$slots,"icon",{class:y(e.cx("icon"))},function(){return[(r(),o(S(e.icon?"span":"ChevronUpIcon"),i({class:[e.cx("icon"),e.icon,u.class]},e.ptm("root").icon,{"data-pc-section":"icon"}),null,16,["class"]))]})]}),_:3},16,["class","onClick","aria-label","unstyled","pt"])):b("",!0)]}),_:3},16,["onEnter","onAfterLeave"])}A.render=P;export{A as default};
