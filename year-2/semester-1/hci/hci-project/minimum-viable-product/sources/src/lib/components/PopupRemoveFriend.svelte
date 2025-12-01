<script lang="ts">
    import Popup from "$lib/components/Popup.svelte";
    import * as AlertDialog from "$lib/shadcn/components/ui/alert-dialog";
    import { buttonVariants } from "$lib/shadcn/components/ui/button";
    import { mutation } from "$lib/utils.svelte";

    let {
        open = $bindable(false),
        user,
        onconfirm
    }: {
        open?: boolean;
        user: { id: string; username: string };
        onconfirm: () => Promise<void>;
    } = $props();

    const mutRemove = mutation(onconfirm);
</script>

<Popup alert bind:open>
    {#snippet trigger()}
        <AlertDialog.Trigger class={[buttonVariants({ variant: "destructive" }), "shadow-md"]}>
            Remove Friend
        </AlertDialog.Trigger>
    {/snippet}
    <AlertDialog.Header>
        <AlertDialog.Title>Are you sure?</AlertDialog.Title>
        <AlertDialog.Description>
            This will remove <em>{user.username}</em>
            from your friends list.
        </AlertDialog.Description>
    </AlertDialog.Header>
    <AlertDialog.Footer>
        <AlertDialog.Cancel class="shadow-md">Cancel</AlertDialog.Cancel>
        <AlertDialog.Action
            class={["shadow-md", mutRemove.applying && "animate-pulse"]}
            disabled={mutRemove.applying}
            onclick={() => (mutRemove.desiredValue = true)}
        >
            Remove
        </AlertDialog.Action>
    </AlertDialog.Footer>
</Popup>
