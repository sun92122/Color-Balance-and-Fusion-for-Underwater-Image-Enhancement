import{u as e}from"./DmfvTtO6.js";import{s as r}from"./CVbYjBld.js";import{I as a,J as n,v as s,x as u,Z as i,a0 as p}from"./DBUFqs3-.js";var d=a`
    .p-radiobutton-group {
        display: inline-flex;
    }
`,m={root:"p-radiobutton-group p-component"},c=n.extend({name:"radiobuttongroup",style:d,classes:m}),l={name:"BaseRadioButtonGroup",extends:r,style:c,provide:function(){return{$pcRadioButtonGroup:this,$parentInstance:this}}},f={name:"RadioButtonGroup",extends:l,inheritAttrs:!1,data:function(){return{groupName:this.name}},watch:{name:function(o){this.groupName=o||e("radiobutton-group-")}},mounted:function(){this.groupName=this.groupName||e("radiobutton-group-")}};function g(t,o,h,v,$,B){return u(),s("div",p({class:t.cx("root")},t.ptmi("root")),[i(t.$slots,"default")],16)}f.render=g;export{f as default};
