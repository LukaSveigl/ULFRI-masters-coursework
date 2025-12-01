<script lang="ts">
    import { enhance } from "$app/forms";
    import Label from "$lib/components/Label.svelte";
    import { Button } from "$lib/shadcn/components/ui/button";
    import { Input } from "$lib/shadcn/components/ui/input";

    let { form } = $props();

    let registrationToken = $state("");

    function onClickRegistrationToken() {
        registrationToken = prompt("Registration token") ?? "";
    }
</script>

<main class="relative flex h-full items-center justify-center">
    <form class="flex flex-col gap-4" method="POST" action="?/login" use:enhance>
        <div>
            <Label label="Username">
                <Input type="text" name="username" />
            </Label>
        </div>
        <div>
            <Label label="Password">
                <Input type="password" name="password" />
            </Label>
        </div>
        {#if registrationToken != ""}
            <input type="hidden" name="registration_token" value={registrationToken} />
        {/if}
        <Button class="mt-5 shadow-md" type="submit">Log In</Button>
        {#if form}
            <p class="text-destructive">{form.message}</p>
        {/if}
    </form>
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div class="absolute right-0 top-0 size-16" onclick={onClickRegistrationToken}></div>
</main>
