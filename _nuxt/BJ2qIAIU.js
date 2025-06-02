import{s as A}from"./BaQveCe0.js";import{s as P}from"./BDGClEDK.js";import k from"./kTh9b-BQ.js";import _ from"./CY_zui-9.js";import I from"./AKnWphcK.js";import{I as C,J as $,a7 as x,a0 as s,g as v,v as f,x as i,Z as B,a1 as S,a2 as w,X as l,B as u,A as b,$ as N,Y as g,a3 as p,z as H}from"./Cfi3JIv2.js";import"./Dq8ienx-.js";import"./Be1fzYNM.js";import"./I-UBfmN-.js";import"./1eywZ9Hy.js";import"./Dshc_X4Y.js";import"./DmfvTtO6.js";var D=C`
    .p-accordionpanel {
        display: flex;
        flex-direction: column;
        border-style: solid;
        border-width: dt('accordion.panel.border.width');
        border-color: dt('accordion.panel.border.color');
    }

    .p-accordionheader {
        all: unset;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: dt('accordion.header.padding');
        color: dt('accordion.header.color');
        background: dt('accordion.header.background');
        border-style: solid;
        border-width: dt('accordion.header.border.width');
        border-color: dt('accordion.header.border.color');
        font-weight: dt('accordion.header.font.weight');
        border-radius: dt('accordion.header.border.radius');
        transition:
            background dt('accordion.transition.duration'),
            color dt('accordion.transition.duration'),
            outline-color dt('accordion.transition.duration'),
            box-shadow dt('accordion.transition.duration');
        outline-color: transparent;
    }

    .p-accordionpanel:first-child > .p-accordionheader {
        border-width: dt('accordion.header.first.border.width');
        border-start-start-radius: dt('accordion.header.first.top.border.radius');
        border-start-end-radius: dt('accordion.header.first.top.border.radius');
    }

    .p-accordionpanel:last-child > .p-accordionheader {
        border-end-start-radius: dt('accordion.header.last.bottom.border.radius');
        border-end-end-radius: dt('accordion.header.last.bottom.border.radius');
    }

    .p-accordionpanel:last-child.p-accordionpanel-active > .p-accordionheader {
        border-end-start-radius: dt('accordion.header.last.active.bottom.border.radius');
        border-end-end-radius: dt('accordion.header.last.active.bottom.border.radius');
    }

    .p-accordionheader-toggle-icon {
        color: dt('accordion.header.toggle.icon.color');
    }

    .p-accordionpanel:not(.p-disabled) .p-accordionheader:focus-visible {
        box-shadow: dt('accordion.header.focus.ring.shadow');
        outline: dt('accordion.header.focus.ring.width') dt('accordion.header.focus.ring.style') dt('accordion.header.focus.ring.color');
        outline-offset: dt('accordion.header.focus.ring.offset');
    }

    .p-accordionpanel:not(.p-accordionpanel-active):not(.p-disabled) > .p-accordionheader:hover {
        background: dt('accordion.header.hover.background');
        color: dt('accordion.header.hover.color');
    }

    .p-accordionpanel:not(.p-accordionpanel-active):not(.p-disabled) .p-accordionheader:hover .p-accordionheader-toggle-icon {
        color: dt('accordion.header.toggle.icon.hover.color');
    }

    .p-accordionpanel:not(.p-disabled).p-accordionpanel-active > .p-accordionheader {
        background: dt('accordion.header.active.background');
        color: dt('accordion.header.active.color');
    }

    .p-accordionpanel:not(.p-disabled).p-accordionpanel-active > .p-accordionheader .p-accordionheader-toggle-icon {
        color: dt('accordion.header.toggle.icon.active.color');
    }

    .p-accordionpanel:not(.p-disabled).p-accordionpanel-active > .p-accordionheader:hover {
        background: dt('accordion.header.active.hover.background');
        color: dt('accordion.header.active.hover.color');
    }

    .p-accordionpanel:not(.p-disabled).p-accordionpanel-active > .p-accordionheader:hover .p-accordionheader-toggle-icon {
        color: dt('accordion.header.toggle.icon.active.hover.color');
    }

    .p-accordioncontent-content {
        border-style: solid;
        border-width: dt('accordion.content.border.width');
        border-color: dt('accordion.content.border.color');
        background-color: dt('accordion.content.background');
        color: dt('accordion.content.color');
        padding: dt('accordion.content.padding');
    }
`,E={root:"p-accordion p-component"},z=$.extend({name:"accordion",style:D,classes:E}),K={name:"BaseAccordion",extends:x,props:{value:{type:[String,Number,Array],default:void 0},multiple:{type:Boolean,default:!1},lazy:{type:Boolean,default:!1},tabindex:{type:Number,default:0},selectOnFocus:{type:Boolean,default:!1},expandIcon:{type:String,default:void 0},collapseIcon:{type:String,default:void 0},activeIndex:{type:[Number,Array],default:null}},style:z,provide:function(){return{$pcAccordion:this,$parentInstance:this}}},F={name:"Accordion",extends:K,inheritAttrs:!1,emits:["update:value","update:activeIndex","tab-open","tab-close","tab-click"],data:function(){return{d_value:this.value}},watch:{value:function(e){this.d_value=e},activeIndex:{immediate:!0,handler:function(e){this.hasAccordionTab&&(this.d_value=this.multiple?e==null?void 0:e.map(String):e==null?void 0:e.toString())}}},methods:{isItemActive:function(e){var o;return this.multiple?(o=this.d_value)===null||o===void 0?void 0:o.includes(e):this.d_value===e},updateValue:function(e){var o,n=this.isItemActive(e);this.multiple?n?this.d_value=this.d_value.filter(function(d){return d!==e}):this.d_value?this.d_value.push(e):this.d_value=[e]:this.d_value=n?null:e,this.$emit("update:value",this.d_value),this.$emit("update:activeIndex",this.multiple?(o=this.d_value)===null||o===void 0?void 0:o.map(Number):Number(this.d_value)),this.$emit(n?"tab-close":"tab-open",{originalEvent:void 0,index:Number(e)})},isAccordionTab:function(e){return e.type.name==="AccordionTab"},getTabProp:function(e,o){return e.props?e.props[o]:void 0},getKey:function(e,o){return this.getTabProp(e,"header")||o},getHeaderPT:function(e,o){var n=this;return{root:s({onClick:function(a){return n.onTabClick(a,o)}},this.getTabProp(e,"headerProps"),this.getTabPT(e,"header",o)),toggleicon:s(this.getTabProp(e,"headeractionprops"),this.getTabPT(e,"headeraction",o))}},getContentPT:function(e,o){return{root:s(this.getTabProp(e,"contentProps"),this.getTabPT(e,"toggleablecontent",o)),transition:this.getTabPT(e,"transition",o),content:this.getTabPT(e,"content",o)}},getTabPT:function(e,o,n){var d=this.tabs.length,a={props:e.props||{},parent:{instance:this,props:this.$props,state:this.$data},context:{index:n,count:d,first:n===0,last:n===d-1,active:this.isItemActive("".concat(n))}};return s(this.ptm("accordiontab.".concat(o),a),this.ptmo(this.getTabProp(e,"pt"),o,a))},onTabClick:function(e,o){this.$emit("tab-click",{originalEvent:e,index:o})}},computed:{tabs:function(){var e=this;return this.$slots.default().reduce(function(o,n){return e.isAccordionTab(n)?o.push(n):n.children&&n.children instanceof Array&&n.children.forEach(function(d){e.isAccordionTab(d)&&o.push(d)}),o},[])},hasAccordionTab:function(){return this.tabs.length}},components:{AccordionPanel:I,AccordionHeader:_,AccordionContent:k,ChevronUpIcon:P,ChevronRightIcon:A}};function U(r,e,o,n,d,a){var m=v("AccordionHeader"),T=v("AccordionContent"),y=v("AccordionPanel");return i(),f("div",s({class:r.cx("root")},r.ptmi("root")),[a.hasAccordionTab?(i(!0),f(S,{key:0},w(a.tabs,function(t,c){return i(),l(y,{key:a.getKey(t,c),value:"".concat(c),pt:{root:a.getTabPT(t,"root",c)},disabled:a.getTabProp(t,"disabled")},{default:u(function(){return[b(m,{class:N(a.getTabProp(t,"headerClass")),pt:a.getHeaderPT(t,c)},{toggleicon:u(function(h){return[h.active?(i(),l(p(r.$slots.collapseicon?r.$slots.collapseicon:r.collapseIcon?"span":"ChevronDownIcon"),s({key:0,class:[r.collapseIcon,h.class],"aria-hidden":"true"},{ref_for:!0},a.getTabPT(t,"headericon",c)),null,16,["class"])):(i(),l(p(r.$slots.expandicon?r.$slots.expandicon:r.expandIcon?"span":"ChevronUpIcon"),s({key:1,class:[r.expandIcon,h.class],"aria-hidden":"true"},{ref_for:!0},a.getTabPT(t,"headericon",c)),null,16,["class"]))]}),default:u(function(){return[t.children&&t.children.headericon?(i(),l(p(t.children.headericon),{key:0,isTabActive:a.isItemActive("".concat(c)),active:a.isItemActive("".concat(c)),index:c},null,8,["isTabActive","active","index"])):g("",!0),t.props&&t.props.header?(i(),f("span",s({key:1,ref_for:!0},a.getTabPT(t,"headertitle",c)),H(t.props.header),17)):g("",!0),t.children&&t.children.header?(i(),l(p(t.children.header),{key:2})):g("",!0)]}),_:2},1032,["class","pt"]),b(T,{pt:a.getContentPT(t,c)},{default:u(function(){return[(i(),l(p(t)))]}),_:2},1032,["pt"])]}),_:2},1032,["value","pt","disabled"])}),128)):B(r.$slots,"default",{key:1})],16)}F.render=U;export{F as default};
