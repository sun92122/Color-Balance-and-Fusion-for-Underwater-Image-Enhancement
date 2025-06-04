import{u as r}from"./DmfvTtO6.js";import{s as t}from"./BiOOapmj.js";import{I as s,J as a,v as n,x as p,Z as c,a0 as u}from"./vUeD5rKf.js";var i=s`
    .p-checkbox-group {
        display: inline-flex;
    }
`,m={root:"p-checkbox-group p-component"},l=a.extend({name:"checkboxgroup",style:i,classes:m}),d={name:"BaseCheckboxGroup",extends:t,style:l,provide:function(){return{$pcCheckboxGroup:this,$parentInstance:this}}},h={name:"CheckboxGroup",extends:d,inheritAttrs:!1,data:function(){return{groupName:this.name}},watch:{name:function(o){this.groupName=o||r("checkbox-group-")}},mounted:function(){this.groupName=this.groupName||r("checkbox-group-")}};function x(e,o,f,k,g,b){return p(),n("div",u({class:e.cx("root")},e.ptmi("root")),[c(e.$slots,"default")],16)}h.render=x;export{h as default};
