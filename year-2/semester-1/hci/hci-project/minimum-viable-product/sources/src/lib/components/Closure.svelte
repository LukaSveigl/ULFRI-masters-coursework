<script module lang="ts">
    import type { Component, ComponentProps } from "svelte";

    export type SvClosure<C extends Component<any, any, any> = Component<any, any, any>> = {
        component: C;
        props: ComponentProps<C>;
    };

    export function mkClosure<C extends Component<any, any, any>>(
        component: C,
        props: ComponentProps<C>
    ): SvClosure<C> {
        return {
            component: component,
            props: props
        };
    }
</script>

<script lang="ts">
    let { this: closure }: { this: SvClosure<any> } = $props();
</script>

<closure.component {...closure.props} />
