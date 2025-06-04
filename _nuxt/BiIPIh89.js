import t from"./F3IPWLaN.js";import{I as o,J as s,g as n,v as l,x as i,Z as d,A as p,a0 as a}from"./vUeD5rKf.js";import"./Be1fzYNM.js";var v=o`
    .p-overlaybadge {
        position: relative;
    }

    .p-overlaybadge .p-badge {
        position: absolute;
        inset-block-start: 0;
        inset-inline-end: 0;
        transform: translate(50%, -50%);
        transform-origin: 100% 0;
        margin: 0;
        outline-width: dt('overlaybadge.outline.width');
        outline-style: solid;
        outline-color: dt('overlaybadge.outline.color');
    }

    .p-overlaybadge .p-badge:dir(rtl) {
        transform: translate(-50%, -50%);
    }
`,g={root:"p-overlaybadge"},c=s.extend({name:"overlaybadge",style:v,classes:g}),m={name:"OverlayBadge",extends:t,style:c,provide:function(){return{$pcOverlayBadge:this,$parentInstance:this}}},y={name:"OverlayBadge",extends:m,inheritAttrs:!1,components:{Badge:t}};function u(e,b,B,f,$,h){var r=n("Badge");return i(),l("div",a({class:e.cx("root")},e.ptmi("root")),[d(e.$slots,"default"),p(r,a(e.$props,{pt:e.ptm("pcBadge")}),null,16,["pt"])],16)}y.render=u;export{y as default};
